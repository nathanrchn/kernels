import torch
import tilelang
import tilelang.language as T

def torch_xielu(x: torch.Tensor, a_p: float = 0.8, a_n: float = 0.8, beta: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    alpha_p = torch.nn.functional.softplus(a_p)
    alpha_n = beta + torch.nn.functional.softplus(a_n)
    return torch.where(x > 0, alpha_p * x * x + beta * x, alpha_n * torch.expm1(torch.clamp_max(x, eps)) - alpha_n * x + beta * x)
