# Copyright (c) OpenMMLab. All rights reserved.
import torch


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.
    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration as e:
        raise ValueError('The input module should contain parameters.') from e

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()

    return torch.device('cpu')