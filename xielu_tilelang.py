import torch
import tilelang
from typing import Tuple
from tvm import DataType
import tilelang.language as T
import tilelang.language.math_intrinsics as M

@tilelang.jit(out_idx=[1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_xielu_forward(N: int, N_CHUNKS: int = 16, threads: int = 256, dtype: str = "bfloat16", activation_dtype: str = "float32"):
    VECTOR_SIZE = 128 // DataType(dtype).bits
    CHUNK = threads * VECTOR_SIZE
    BLOCK_N = N_CHUNKS * CHUNK

    @T.prim_func
    def main(X: T.Tensor((N), dtype), Y: T.Tensor((N), dtype), s_a_p: T.float32, s_a_n: T.float32, beta: T.float32 = 0.5, eps: T.float32 = 1e-6):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=threads) as (bn):
            tn = T.get_thread_binding(0)
            X_local = T.alloc_local((VECTOR_SIZE), dtype=dtype)
            Y_local = T.alloc_local((VECTOR_SIZE), dtype=dtype)

            for c in T.serial(N_CHUNKS):
                base = bn * BLOCK_N + c * CHUNK
                for j in T.vectorized(VECTOR_SIZE):
                    X_local[j] = X[base + tn * VECTOR_SIZE + j]

                for i in T.serial(VECTOR_SIZE):
                    x = T.Cast(activation_dtype, X_local[i])
                    y = T.if_then_else(x > 0.0, x * M.ieee_fmaf(s_a_p, x, beta), M.ieee_fmaf(beta + s_a_n, T.exp(T.min(x, eps)) - 1.0, - s_a_n * x))
                    Y_local[i] = T.Cast(dtype, y)

                for j in T.vectorized(VECTOR_SIZE):
                    Y[base + tn * VECTOR_SIZE + j] = Y_local[j]

    return main

@tilelang.jit(out_idx=[], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_xielu_backward(N: int, N_CHUNKS: int = 16, threads: int = 256, dtype: str = "bfloat16", activation_dtype: str = "float32"):
    VECTOR_SIZE = 128 // DataType(dtype).bits
    CHUNK = threads * VECTOR_SIZE
    BLOCK_N = N_CHUNKS * CHUNK

    @T.prim_func
    def main(GO: T.Tensor((N), dtype), X: T.Tensor((N), dtype), GI: T.Tensor((N), dtype), da_p: T.Tensor((1), activation_dtype), da_n: T.Tensor((1), activation_dtype), s_a_p: T.float32, ds_a_p: T.float32, s_a_n: T.float32, ds_a_n: T.float32, beta: T.float32 = 0.5, eps: T.float32 = 1e-6):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=threads) as (bn):
            tn = T.get_thread_binding(0)
            X_local = T.alloc_local((VECTOR_SIZE), dtype=dtype)
            GO_local = T.alloc_local((VECTOR_SIZE), dtype=dtype)
            GI_local = T.alloc_local((VECTOR_SIZE), dtype=dtype)
            DAP_local = T.alloc_local((1), dtype=activation_dtype)
            DAN_local = T.alloc_local((1), dtype=activation_dtype)
            DAP_local[0] = 0.0
            DAN_local[0] = 0.0

            alpha_n = beta + s_a_n
            for c in T.serial(N_CHUNKS):
                base = bn * BLOCK_N + c * CHUNK
                for j in T.vectorized(VECTOR_SIZE):
                    X_local[j] = X[base + tn * VECTOR_SIZE + j]
                    GO_local[j] = GO[base + tn * VECTOR_SIZE + j]

                for j in T.serial(VECTOR_SIZE):
                    x = T.Cast(activation_dtype, X_local[j])
                    go = T.Cast(activation_dtype, GO_local[j])
                    dx = T.if_then_else(x > 0.0, go * (2.0 * s_a_p * x + beta), go * (alpha_n * (T.exp(T.min(x, eps)) * T.if_then_else(x <= eps, 1.0, 0.0) - 1.0) + beta))
                    GI_local[j] = T.Cast(dtype, dx)

                    DAP_local[0] += T.if_then_else(x > 0.0, go * ds_a_p * x * x, 0.0)
                    DAN_local[0] += T.if_then_else(x <= 0.0, go * ds_a_n * ((T.exp(T.min(x, eps)) - 1.0) - x), 0.0)

                for j in T.vectorized(VECTOR_SIZE):
                    GI[base + tn * VECTOR_SIZE + j] = GI_local[j]


            for o in T.serial(5):
                DAP_local[0] += T.tvm_warp_shuffle_down(0xffffffff, DAP_local[0], 16 >> o, 32, 32)
                DAN_local[0] += T.tvm_warp_shuffle_down(0xffffffff, DAN_local[0], 16 >> o, 32, 32)
            
            if tn % 32 == 0:
                T.atomic_add(da_p[0], DAP_local[0])
                T.atomic_add(da_n[0], DAN_local[0])

    return main


@torch.library.custom_op("swissai::xielu", mutates_args=())
def xielu(x: torch.Tensor, a_p: torch.Tensor, a_n: torch.Tensor, beta: float, eps: float) -> Tuple[torch.Tensor, float, float]:
    shape = x.shape
    s_a_p = torch.nn.functional.softplus(a_p).item()
    s_a_n = torch.nn.functional.softplus(a_n).item()
    kernel = tilelang_xielu_forward(x.numel(), N_CHUNKS=4, threads=256, dtype=str(x.dtype).split(".")[-1])
    return kernel(x.contiguous().view(-1), s_a_p, s_a_n, beta, eps).view(shape), s_a_p, s_a_n

def xielu_backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
    x, a_p, a_n = ctx.saved_tensors
    ds_a_p = torch.nn.functional.sigmoid(a_p).item()
    ds_a_n = torch.nn.functional.sigmoid(a_n).item()

    gi = torch.empty_like(x)
    da_p = torch.tensor((0.0,), device="cuda")
    da_n = torch.tensor((0.0,), device="cuda")
    kernel = tilelang_xielu_backward(x.numel(), N_CHUNKS=16, threads=256, dtype=str(x.dtype).split(".")[-1])
    kernel(grad_output.contiguous().view(-1), x.contiguous().view(-1), gi.contiguous().view(-1), da_p, da_n, ctx.s_a_p, ds_a_p, ctx.s_a_n, ds_a_n, ctx.beta, ctx.eps)
    return gi.view(x.shape), da_p, da_n, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    x, a_p, a_n, beta, eps = inputs
    ctx.save_for_backward(x, a_p, a_n)
    ctx.beta = beta
    ctx.eps = eps
    _, s_a_p, s_a_n = output
    ctx.s_a_p = s_a_p
    ctx.s_a_n = s_a_n

xielu.register_autograd(xielu_backward, setup_context=setup_context)
