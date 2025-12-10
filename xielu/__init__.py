import torch
from torch import Tensor

import _xielu  # noqa: F401


@torch.library.register_fake("xielu::forward")
def _(x: Tensor, alpha_p: Tensor, alpha_n: Tensor, beta: float, eps: float) -> Tensor:
    return torch.empty_like(x)


@torch.library.register_fake("xielu::backward")
def _(
    x: Tensor, go: Tensor, alpha_p: Tensor, alpha_n: Tensor, beta: float, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(x),
        torch.empty(1, dtype=torch.float32, device=x.device),
        torch.empty(1, dtype=torch.float32, device=x.device),
    )


class XiELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: Tensor, alpha_p: Tensor, alpha_n: Tensor, beta: float, eps: float
    ) -> Tensor:
        ctx.save_for_backward(x, alpha_p, alpha_n)
        ctx.beta = beta
        ctx.eps = eps
        return torch.ops.xielu.forward(x, alpha_p, alpha_n, beta, eps)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, alpha_p, alpha_n = ctx.saved_tensors
        go = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        x = x.contiguous() if not x.is_contiguous() else x
        gi, galpha_p, galpha_n = torch.ops.xielu.backward(
            x, go, alpha_p, alpha_n, ctx.beta, ctx.eps
        )
        return gi, galpha_p, galpha_n, None, None


def xielu(
    x: Tensor, alpha_p: Tensor, alpha_n: Tensor, beta: float = 1.0, eps: float = -10.0
) -> Tensor:
    return XiELUFunction.apply(x, alpha_p, alpha_n, beta, eps)
