"""CPU inputs for the existing, CPU-oriented MRoPE position builder.

Move tensor arguments once per call, keep the original position arithmetic,
and return both outputs to input_ids.device. Installation is explicit.
"""

from functools import wraps
from inspect import signature

import torch


def _cpu_inputs(original):
    parameters = signature(original)

    @wraps(original)
    def get_rope_index(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        input_ids = bound.arguments.get("input_ids")
        if input_ids is None:
            return original(*args, **kwargs)

        device = input_ids.device
        for name, value in bound.arguments.items():
            if isinstance(value, torch.Tensor):
                bound.arguments[name] = value.cpu()
        positions, deltas = original(*bound.args, **bound.kwargs)
        return positions.to(device), deltas.to(device)

    return get_rope_index


_installed = False


def install():
    global _installed
    if not _installed:
        from slime_ext.models.qwen3_omni import thinker

        thinker.get_rope_index = _cpu_inputs(thinker.get_rope_index)
        _installed = True
