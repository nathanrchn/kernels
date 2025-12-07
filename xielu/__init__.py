import torch
import torch.library

import xielu

@torch.library.register_fake("xielu::forward")
def _(x, alpha_p, alpha_n, beta, eps):
    return torch.empty_like(x)

def backward(ctx, grad_output):
    x, alpha_p, alpha_n = ctx.saved_tensors
    beta, eps = ctx.beta, ctx.eps
    go = grad_output.contiguous()
    x_contig = x.contiguous()
    gi, galpha_p, galpha_n = torch.ops.xielu.backward(x_contig, go, alpha_p, alpha_n, beta, eps)
    return gi, galpha_p.to(alpha_p.dtype), galpha_n.to(alpha_n.dtype), None, None

def setup_context(ctx, inputs, output):
    x, alpha_p, alpha_n, beta, eps = inputs
    ctx.save_for_backward(x, alpha_p, alpha_n)
    ctx.beta = beta
    ctx.eps = eps

torch.library.register_autograd("xielu::forward", backward, setup_context=setup_context)

def xielu(x: torch.Tensor, alpha_p: torch.Tensor, alpha_n: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    return torch.ops.xielu.forward(x, alpha_p, alpha_n, beta, eps)
