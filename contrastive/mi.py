import math

import torch
from torch.distributions.utils import broadcast_all

import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand


class MutualInformation(object):
    def __init__(self, num_outer_samples, data_source=None):
        self.data_source = data_source
        self.num_outer_samples = num_outer_samples

    def _vectorized(self, fn, *shape, name="vectorization_plate"):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        MI computation over `num_particles`, and to broadcast batch shapes of
        sample site functions in accordance with the `~pyro.plate` contexts
        within which they are embedded.

        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, shape):
                return fn(*args, **kwargs)

        return wrapped_fn

    def get_primary_rollout(self, model, args, kwargs, graph_type="flat", detach=False):
        if self.data_source is None:
            model = self._vectorized(
                model, self.num_outer_samples, name="outer_vectorization"
            )
        else:
            data = next(self.data_source)
            model = pyro.condition(
                self._vectorized(
                    model, self.num_outer_samples, name="outer_vectorization"
                ),
                data=data,
            )

        trace = poutine.trace(model, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()
        return trace


class PriorContrastiveEstimation(MutualInformation):
    def __init__(self, num_outer_samples=100, num_inner_samples=10, data_source=None):
        super().__init__(num_outer_samples, data_source=data_source)
        self.num_inner_samples = num_inner_samples

    def compute_observation_log_prob(self, trace):
        """
        Computes the log probability of observations given latent variables and designs.

        :param trace: a Pyro trace object
        :return: the log prob tensor
        """
        return sum(
            node["log_prob"]
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        )

    def get_contrastive_rollout(
        self,
        trace,
        model,
        args,
        kwargs,
        existing_vectorization,
        graph_type="flat",
        detach=False,
    ):
        sampled_observation_values = {
            name: lexpand(node["value"], self.num_inner_samples)
            for name, node in trace.nodes.items()
            if node.get("subtype") in ["observation_sample", "design_sample"]
        }
        conditional_model = self._vectorized(
            pyro.condition(model, data=sampled_observation_values),
            self.num_inner_samples,
            *existing_vectorization,
            name="inner_vectorization",
        )
        trace = poutine.trace(conditional_model, graph_type=graph_type).get_trace(
            *args, **kwargs
        )
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()

        return trace

    def differentiable_loss(self, model, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)

        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)

        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float

        Evaluates the MI lower bound using prior contrastive estimation; == -EIG proxy
        """
        loss_to_constant = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_to_constant - math.log(self.num_inner_samples + 1)


class NestedMonteCarloEstimation(PriorContrastiveEstimation):
    def differentiable_loss(self, model, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)
        obs_log_prob_combined = obs_log_prob_contrastive.logsumexp(0)
        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float

        Evaluates the MI lower bound using prior contrastive estimation
        """
        loss_to_constant = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_to_constant - math.log(self.num_inner_samples)



class AdaptiveContrastiveEstimation(PriorContrastiveEstimation):

    def compute_latent_log_prob(self, trace):
        """
        Computes the log probability of latent variables in the model.

        :param trace: a Pyro trace object
        :return: the log prob tensor
        """
        return sum(
            node["log_prob"]
            for node in trace.nodes.values()
            if node.get("subtype") == "latent_sample"
        )

    def get_contrastive_rollout(
        self,
        trace,
        model,
        guide,
        args,
        kwargs,
        existing_vectorization,
        graph_type="flat",
        detach=False,
    ):
        sampled_observation_values = {
            name: lexpand(node["value"], self.num_inner_samples)
            for name, node in trace.nodes.items()
            if node.get("subtype") in ["observation_sample", "design_sample"]
        }
        conditional_guide = self._vectorized(
            pyro.condition(guide, data=sampled_observation_values),
            self.num_inner_samples,
            *existing_vectorization,
            name="inner_vectorization",
        )
        guide_trace = poutine.trace(conditional_guide, graph_type=graph_type).get_trace(
            *args, **kwargs
        )

        contrastive_latent_samples = {
            name: node["value"]
            for name, node in guide_trace.nodes.items()
            if node.get("subtype") == "latent_sample"
        }
        contrastive_latent_samples.update(sampled_observation_values)
        conditional_model = self._vectorized(
            pyro.condition(model, data=contrastive_latent_samples),
            self.num_inner_samples,
            *existing_vectorization,
            name="inner_vectorization",
        )
        model_trace = poutine.trace(conditional_model, graph_type=graph_type).get_trace(
            *args, **kwargs
        )

        if detach:
            guide_trace.detach_()
            model_trace.detach_()
        guide_trace = prune_subsample_sites(guide_trace).compute_log_prob()
        model_trace = prune_subsample_sites(model_trace).compute_log_prob()

        return guide_trace, model_trace


    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_guide_trace, contrastive_model_trace = self.get_contrastive_rollout(
            primary_trace, model, guide, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_model_trace)
        latent_log_prob_prior = self.compute_latent_log_prob(contrastive_model_trace)
        latent_log_prob_guide = self.compute_latent_log_prob(contrastive_guide_trace)

        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)

        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float

        Evaluates the MI lower bound using adaptive contrastive estimation
        """
        loss_to_constant = torch_item(self.differentiable_loss(model, guide, *args, **kwargs))
        return loss_to_constant - math.log(self.num_inner_samples + 1)
