

import torch
import torch.nn as nn
import torch.jit as jit

from typing import Any

from libmlu_ext import cmul_element

class mul_element_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        z = cmul_element(x, y)
        ctx.save_for_backward(*[x, y])
        return z

    @staticmethod
    def backward(ctx: Any, d_r: Any) -> Any:
        d_r = d_r.contiguous()
        x, y = ctx.saved_variables
        d_x = cmul_element(d_r, y)
        d_y = cmul_element(d_r, x)
        return d_x, d_y

@jit.ignore
def mul_element(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return mul_element_function.apply(x, y)

