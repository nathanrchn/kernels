import torch
import tilelang
import tilelang.language as T
import tilelang.language.math_intrinsics as M

def torch_xielu(x: torch.Tensor, a_p: float = 0.8, a_n: float = 0.8, beta: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    a_p = torch.tensor(a_p, dtype=x.dtype, device=x.device)
    a_n = torch.tensor(a_n, dtype=x.dtype, device=x.device)
    beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
    eps = torch.tensor(eps, dtype=x.dtype, device=x.device)
    
    alpha_p = torch.nn.functional.softplus(a_p)
    alpha_n = beta + torch.nn.functional.softplus(a_n)
    return torch.where(x > 0, alpha_p * x * x + beta * x, alpha_n * torch.expm1(torch.clamp_max(x, eps)) - alpha_n * x + beta * x)

def torch_xielu_with_grad(x: torch.Tensor, a_p: torch.Tensor, a_n: torch.Tensor, beta: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Version of torch_xielu that accepts tensor parameters for gradient computation."""
    alpha_p = torch.nn.functional.softplus(a_p)
    alpha_n = beta + torch.nn.functional.softplus(a_n)
    return torch.where(x > 0, alpha_p * x * x + beta * x, alpha_n * torch.expm1(torch.clamp_max(x, eps)) - alpha_n * x + beta * x)

def log1pf(x):
    e_int = (T.Cast("int32", T.reinterpret("uint32", x + 1.0)) - T.Cast("int32", T.reinterpret("uint32", M.ieee_fmaf(x, 0.0, 0.75)))) & 0xff800000
    e = T.Cast("float32", e_int)
    m = T.reinterpret("float32", T.Cast("uint32", T.Cast("int32", T.reinterpret("uint32", x)) - e_int))
    s_prime = T.reinterpret("float32", T.Cast("uint32", T.Cast("int32", T.reinterpret("uint32", M.ieee_fmaf(x, 0.0, 4.0))) - e_int))
    m = m + M.ieee_fmaf(M.ieee_fmaf(x, 0.0, 0.25), s_prime, M.ieee_fmaf(x, 0.0, -1.0))
    s = m * m
    r, t = -4.54559326e-2, 1.05529785e-1
    r = M.ieee_fmaf(r, s, -1.32279143e-1)
    t = M.ieee_fmaf(t, s, 1.44911006e-1)
    r = M.ieee_fmaf(r, s, -1.66416913e-1)
    t = M.ieee_fmaf(t, s, 1.99886635e-1)
    r = M.ieee_fmaf(r, s, -2.50001878e-1)
    r = M.ieee_fmaf(M.ieee_fmaf(M.ieee_fmaf(M.ieee_fmaf(t, m, r), m, 3.33335280e-1), m, -5.00000000e-1), s, m)
    r = M.ieee_fmaf(e, 0.693147182 * 1.19209290e-7, r)
    is_normal = (x != 0.0) & (x > 1.0) & (x < float("inf"))
    special_result = M.ieee_fmaf(T.log2(x + 1.0), 0.693147182, 0.0, rounding_mode="rd")
    return T.if_then_else(is_normal, r, special_result)

def expm1f(a):
    j = M.ieee_fmaf(1.442695, a, 12582912.0)
    i_bits = T.Cast("int32", T.reinterpret("uint32", j))
    j = j - 12582912.0
    f = M.ieee_fmaf(j, -1.42860677e-6, M.ieee_fmaf(j, -6.93145752e-1, a))
    s = f * f
    s = T.if_then_else(a == 0.0, a, s)
    r = M.ieee_fmaf(1.98662281e-4, f, 1.39354519e-3)
    r = M.ieee_fmaf(r, s, M.ieee_fmaf(8.33332818e-3, f, 4.16667648e-2))
    r = M.ieee_fmaf(r, f, 1.66666716e-1)
    r = M.ieee_fmaf(r, f, 4.99999970e-1)
    s_half_b = M.ieee_fmaf(a, 0.0, 0.5)
    u = T.if_then_else(j == 1.0, f + s_half_b, f)
    v = M.ieee_fmaf(r, s, u)
    t_bits = (i_bits << 23) + T.Cast("int32", T.reinterpret("uint32", s_half_b))
    t = T.reinterpret("float32", T.Cast("uint32", t_bits))
    y = t - s_half_b
    res = M.ieee_fmaf(v, t, t - y - s_half_b) + y
    res = T.if_then_else(j == 1.0, v + v, T.if_then_else(j == 0.0, v, res + res))
    return res

def softplus_forward(x):
    return T.if_then_else(x > 20.0, x, T.if_then_else(x < -20.0, 0.0, log1pf(T.exp(x))))

def softplus_backward(x):
    return T.if_then_else(x > 20.0, 1.0, T.if_then_else(x < -20.0, 0.0, 1.0 / (1.0 + T.exp(-x))))

@tilelang.jit(out_idx=[1])
def tilelang_xielu_forward(N: int, threads: int = 256, dtype: str = "bfloat16"):
    @T.prim_func
    def main(X: T.Tensor((N), dtype), Y: T.Tensor((N), dtype), a_p: T.Tensor((1)), a_n: T.Tensor((1)), beta: T.Tensor((1)), eps: T.Tensor((1))):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as (b_x):
            for i in T.Parallel(threads):
                s_a_p = softplus_forward(a_p[0])
                s_a_n = softplus_forward(a_n[0])
                x = T.Cast("float32", X[b_x * threads + i])
                y = T.if_then_else(x > 0.0, x * (s_a_p * x + beta[0]), (beta[0] + s_a_n) * expm1f(T.min(x, eps[0])) - s_a_n * x)
                Y[b_x * threads + i] = T.Cast(dtype, y)

    return main

@tilelang.jit(out_idx=[2, 3, 4])
def tilelang_xielu_backward(N: int, threads: int = 256, dtype: str = "bfloat16"):
    @T.prim_func
    def main(GO: T.Tensor((N), dtype), X: T.Tensor((N), dtype), GI: T.Tensor((N), dtype), da_p: T.Tensor((1)), da_n: T.Tensor((1)), a_p: T.Tensor((1)), a_n: T.Tensor((1)), beta: T.Tensor((1)), eps: T.Tensor((1))):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as (b_x):
            for i in T.Parallel(threads):
                s_a_p = softplus_forward(a_p[0])
                s_a_n = softplus_forward(a_n[0])
                ds_a_p = softplus_backward(a_p[0])
                ds_a_n = softplus_backward(a_n[0])
                x = T.Cast("float32", X[b_x * threads + i])
                go = T.Cast("float32", GO[b_x * threads + i])
                dx = T.if_then_else(x > 0.0, go * (2.0 * s_a_p * x + beta[0]), go * ((beta[0] + s_a_n) * T.exp(T.min(x, eps[0])) - s_a_n))
                GI[b_x * threads + i] = T.Cast(dtype, dx)

                dp = T.if_then_else(x > 0.0, go * ds_a_p * x * x, 0.0)
                dn = T.if_then_else(x <= 0.0, go * ds_a_n * (expm1f(T.min(x, eps[0])) - x), 0.0)

                T.atomic_add(da_p[0], dp)
                T.atomic_add(da_n[0], dn)

    return main
