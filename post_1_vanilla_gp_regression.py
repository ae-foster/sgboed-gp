import numpy as np
import matplotlib.pyplot as plt

import torch
from pyro.infer.util import torch_item
import pyro
import pyro.distributions as dist
from tqdm import trange

from oed.primitives import observation_sample, compute_design, latent_sample
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimation


class RBFGaussianProcessModel:
    """Implements a Pyro GP model suitable for SGBOED."""

    def __init__(self, batch_size=6, dim=2, lengthscale=1.0, device="cuda:0"):
        self.batch_size = batch_size
        self.dim = dim
        self.lengthscale = lengthscale
        self.device = device

    def model(self):
        # The design is a pyro.param that is tanh transformer to [-1, 1]^{dim}
        design = pyro.param("design", torch.empty([self.batch_size, self.dim], device=self.device).normal_())
        design = torch.tanh(design)

        # Evaluate the RBF kernel between each design point in the batch design
        cov = torch.exp(-(design.unsqueeze(-3) - design.unsqueeze(-2)).pow(2).sum(-1) / (2 * self.lengthscale ** 2))
        # Add tiny diagonal variance for stability. This is *not* observation noise
        cov = cov + 1e-7 * torch.eye(cov.shape[-1])

        # The `latent_sample` instruction overrides pyro.sample and tags the sample as a model latent
        means = latent_sample("means", dist.MultivariateNormal(torch.zeros(self.batch_size, device=self.device), cov))
        # The `observation_sample` instruction similarly adds a tag to this sample state for an experimental outcome
        # The observation variance is set to 1.0
        # The `.to_event(1)` tells Pyro that y is vector of dimension `self.batch_size`
        y = observation_sample("y", dist.Normal(means, 1.0).to_event(1))
        return y


def single_run(seed, num_steps=20000):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)

    pce_loss = PriorContrastiveEstimation(10, 100)
    gp_model = RBFGaussianProcessModel(batch_size=12, device="cpu")

    optimizer = pyro.optim.Adam(optim_args={"lr": 0.001, "weight_decay": 0})
    oed = OED(gp_model.model, optimizer, pce_loss)

    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    for i in t:
        loss = oed.step()
        loss = torch_item(loss)
        # The loss here is not quite PCE, it is log(L+1) - PCE
        # Thus, lower loss => higher PCE, loss = 0 implies PCE is maximised at log(L+1)
        t.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 2000 == 1:
            d = np.tanh(pyro.param("design").cpu().detach().numpy())
            plt.scatter(d[:, 0], d[:, 1])
            plt.savefig(f"design_{i}.png")
            plt.close()

    # Make some additional plots
    d = np.tanh(pyro.param("design").cpu().detach().numpy())
    plt.scatter(d[:, 0], d[:, 1])
    plt.savefig("design_end.png")
    plt.close()

    plt.plot(loss_history)
    plt.xlabel("Step")
    plt.ylabel("log(L+1) - PCE")
    plt.savefig("loss_history.png")
    plt.close()


if __name__ == "__main__":
    seed = np.random.randint(100000)
    single_run(seed)
