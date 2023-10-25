

import torch
import torch.nn as nn
import torch.jit as jit

from typing import Any

from libmlu_ext import cmatmul

class matmul_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, x_scale=1.0, y_scale=1.0):
        z = cmatmul(x, y, x_scale, y_scale)
        ctx.save_for_backward(*[x, y, x_scale, y_scale])
        return z

    @staticmethod
    def backward(ctx: Any, d_r: Any) -> Any:
        d_r = d_r.contiguous()
        x, y, x_scale, y_scale = ctx.saved_variables
        d_x = cmatmul(d_r, y.transpose(1,0))
        d_y = cmatmul(x.transpose(1,0), d_r)
        return d_x, d_y

@jit.ignore
def cmm(x: torch.Tensor, y: torch.Tensor, x_scale=1.0, y_scale=1.0) -> torch.Tensor:
    return matmul_function.apply(x, y, x_scale, y_scale)

