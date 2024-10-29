import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from contrastive.mi import PriorContrastiveEstimation
from oed.design import OED
from oed.primitives import latent_sample, observation_sample
from pyro.infer.util import torch_item
from tqdm import trange


class PeriodicGaussianProcessModel:
    """Implements a Pyro Periodic GP model with a known period, suitable for SGBOED."""

    def __init__(self, batch_size=6, period=1.0, device="cuda:0"):
        self.batch_size = batch_size
        self.period = period
        self.device = device

    def model(self):
        # The design is a pyro.param that lives on the real line
        design = pyro.param(
            "design", 0.5 * torch.empty([self.batch_size], device=self.device).uniform_()
        )

        # Evaluate the periodic kernel between each design point in the batch design
        d = torch.abs(design.unsqueeze(-2) - design.unsqueeze(-1))
        cov = torch.exp(-torch.sin(d * torch.pi / self.period).pow(2))
        # Add tiny diagonal variance for stability. This is *not* observation noise
        cov = cov + 1e-7 * torch.eye(cov.shape[-1])

        # The `latent_sample` instruction overrides pyro.sample and tags the sample as a model latent
        means = latent_sample(
            "means", dist.MultivariateNormal(torch.zeros(self.batch_size, device=self.device), cov)
        )
        # The `observation_sample` instruction similarly adds a tag to this sample site for an experimental outcome
        # The observation variance is set to 1.0

        # The `.to_event(1)` tells Pyro that y is vector of dimension `self.batch_size`
        # By adding `.to_event(1)`, when Pyro computes the conditional log-likelihood of y, it sums over the individual
        #  batch components of `y`, which is correct, rather than returning a separate log-likelihood for each
        #  component of `y`, which would give an incorrectly shaped likelihood tensor
        y = observation_sample("y", dist.Normal(means, 1.0).to_event(1))
        return y


def single_run(seed, num_steps=20000):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)

    pce_loss = PriorContrastiveEstimation(10, 100)
    gp_model = PeriodicGaussianProcessModel(
        batch_size=9, device="cpu"
    )  # small tensors may be faster on CPU

    optimizer = pyro.optim.Adam(optim_args={"lr": 0.001, "weight_decay": 0, "betas": (0.5, 0.9)})
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
            d = pyro.param("design").cpu().detach().numpy()
            d = np.sort(d)
            x = torch.arange(gp_model.batch_size)
            plt.scatter(x, d)
            plt.savefig(f"design_{i}.png")
            plt.close()

    # Make some additional plots
    d = pyro.param("design").cpu().detach().numpy()
    d = np.sort(d)
    x = torch.arange(gp_model.batch_size)
    plt.scatter(x, d)
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
