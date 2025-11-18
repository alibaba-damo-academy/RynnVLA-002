import torch
import torch.nn as nn


def promote_param_to_fp32(param: nn.Parameter) -> None:
    if param.is_floating_point() and torch.finfo(param.dtype).bits < 32:
        param.data = param.data.float()
    if param.is_complex() and torch.finfo(param.dtype).bits < 32:
        param.data = param.data.to(torch.complex64)
