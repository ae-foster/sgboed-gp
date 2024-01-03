from typing import Optional, Dict, Union
import warnings
import torch

from pyro.poutine.runtime import am_i_wrapped, apply_stack
from pyro.distributions import TorchDistribution


def sample_with_type(type_string, name, fn, *args, **kwargs):
    """
    Over-ride Pyro's sample primitive to add a subtype marker.
    """
    # Transform obs_mask into multiple sample statements.
    obs = kwargs.pop("obs", None)
    obs_mask = kwargs.pop("obs_mask", None)
    if obs_mask is not None:
        return _masked_observe(name, fn, obs, obs_mask, *args, **kwargs)

    # Check if stack is empty.
    # if stack empty, default behavior (defined here)
    infer = kwargs.pop("infer", {}).copy()
    is_observed = infer.pop("is_observed", obs is not None)
    if not am_i_wrapped():
        if obs is not None and not infer.get("_deterministic"):
            warnings.warn(
                "trying to observe a value outside of inference at " + name,
                RuntimeWarning,
            )
            return obs
        return fn(*args, **kwargs)
    # if stack not empty, apply everything in the stack?
    else:
        # initialize data structure to pass up/down the stack
        msg = {
            "type": "sample",
            "subtype": type_string,
            "name": name,
            "fn": fn,
            "is_observed": is_observed,
            "args": args,
            "kwargs": kwargs,
            "value": obs,
            "infer": infer,
            "scale": 1.0,
            "mask": None,
            "cond_indep_stack": (),
            "done": False,
            "stop": False,
            "continuation": None,
        }
        # apply the stack and return its return value
        apply_stack(msg)
        return msg["value"]


def observation_sample(name, fn, *args, **kwargs):
    return sample_with_type("observation_sample", name, fn, *args, **kwargs)


def compute_design(name, fn, *args, **kwargs):
    return sample_with_type("design_sample", name, fn, *args, **kwargs)


def latent_sample(name, fn, *args, **kwargs):
    return sample_with_type("latent_sample", name, fn, *args, **kwargs)
