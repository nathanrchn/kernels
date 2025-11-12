import torch
from torch.utils.cpp_extension import load_inline

cuda_src = """
#include <stdio.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <torch/library.h>

constexpr int N_THREADS = 256;
constexpr int FORWARD_BLOCK_SIZE = 8192;
constexpr int FORWARD_CHUNK_SIZE = 2048;

constexpr int BACKWARD_BLOCK_SIZE = 32768;
constexpr int BACKWARD_CHUNK_SIZE = 2048;

__global__ void xielu_forward_kernel(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, float s_a_p, float s_a_n, float beta, float eps, int N) {
    __nv_bfloat16 x_local[8];
    __nv_bfloat16 y_local[8];

    for (int i = 0; i < 4; i++) {
        *(uint4*)x_local = *(uint4*)(x + (int)blockIdx.x * FORWARD_BLOCK_SIZE + i * FORWARD_CHUNK_SIZE + (int)threadIdx.x * 8);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float x = __bfloat162float(x_local[j]);
            float y = (x > 0.0f) ? x * __fmaf_rn(s_a_p, x, beta) : __fmaf_rn(beta + s_a_n, __expf(fminf(x, eps)) - 1.0f, -s_a_n * x);
            y_local[j] = __float2bfloat16(y);
        }

        *(uint4*)(y + (int)blockIdx.x * FORWARD_BLOCK_SIZE + i * FORWARD_CHUNK_SIZE + (int)threadIdx.x * 8) = *(uint4*)y_local;
    }
}

__global__ void xielu_backward_kernel(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ go, __nv_bfloat16* __restrict__ gi, float* __restrict__ da_p, float* __restrict__ da_n, float s_a_p, float s_a_n, float ds_a_p, float ds_a_n, float beta, float eps, int N) {
    __nv_bfloat16 x_local[8];
    __nv_bfloat16 go_local[8];
    __nv_bfloat16 gi_local[8];
    float da_p_local[1] = {0.0};
    float da_n_local[1] = {0.0};

    const float two_s_a_p = 2.0f * s_a_p;
    const float alpha_n = beta + s_a_n;
    for (int i = 0; i < 16; i++) {
        *(uint4*)x_local = *(uint4*)(x + (int)blockIdx.x * BACKWARD_BLOCK_SIZE + i * BACKWARD_CHUNK_SIZE + (int)threadIdx.x * 8);
        *(uint4*)go_local = *(uint4*)(go + (int)blockIdx.x * BACKWARD_BLOCK_SIZE + i * BACKWARD_CHUNK_SIZE + (int)threadIdx.x * 8);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float x = __bfloat162float(x_local[j]);
            float go = __bfloat162float(go_local[j]);

            float e = __expf(fminf(x, eps));
            float below_eps = (x <= eps) ? 1.0f : 0.0f;
            float dx = (x > 0.0f) ? go * (two_s_a_p * x + beta) : go * (alpha_n * (e * below_eps - 1.0f) + beta);
            gi_local[j] = __float2bfloat16(dx);

            da_p_local[0] += (x > 0.0f) ? go * ds_a_p * x * x : 0.0f;
            da_n_local[0] += (x <= 0.0f) ? go * ds_a_n * ((e - 1.0f) - x) : 0.0f;
        }

        *(uint4*)(gi + (int)blockIdx.x * BACKWARD_BLOCK_SIZE + i * BACKWARD_CHUNK_SIZE + (int)threadIdx.x * 8) = *(uint4*)gi_local;
    }

    for (int i = 16; i > 0; i /= 2) {
        da_p_local[0] += __shfl_down_sync(0xffffffff, da_p_local[0], i, 32);
        da_n_local[0] += __shfl_down_sync(0xffffffff, da_n_local[0], i, 32);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(da_p, da_p_local[0]);
        atomicAdd(da_n, da_n_local[0]);
    }
}

float softplus_forward(float x) {
    return (x > 20.0) ? x : ((x < -20.0) ? 0.0 : log1pf(exp(x)));
}

float softplus_backward(float x) {
    return (x > 20.0) ? 1.0 : ((x < -20.0) ? 0.0 : 1.0 / (1.0 + exp(-x)));
}

at::Tensor xielu_forward(const at::Tensor &x, const at::Tensor &alphas, double beta, double eps) {
    TORCH_CHECK(x.dtype() == at::kBFloat16);
    TORCH_CHECK(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alphas.dtype() == at::kFloat);
    TORCH_CHECK(alphas.numel() == 2);
    TORCH_CHECK(alphas.device().type() == at::DeviceType::CPU);
    TORCH_CHECK(beta > 0.0);
    TORCH_CHECK(eps > 0.0);

    at::Tensor y = at::empty(x.sizes(), x.options());

    __nv_bfloat16* x_ptr = reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16* y_ptr = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());

    float* alphas_ptr = alphas.data_ptr<float>();
    float s_a_p = softplus_forward(alphas_ptr[0]);
    float s_a_n = softplus_forward(alphas_ptr[1]);

    int N = x.numel();
    int n_blocks = (N + FORWARD_BLOCK_SIZE - 1) / FORWARD_BLOCK_SIZE;
    
    xielu_forward_kernel<<<n_blocks, N_THREADS>>>(x_ptr, y_ptr, s_a_p, s_a_n, static_cast<float>(beta), static_cast<float>(eps), N);

    return y;
}

at::Tensor xielu_backward(const at::Tensor &x, const at::Tensor &go, const at::Tensor &alphas, double beta, double eps) {
    TORCH_CHECK(x.dtype() == at::kBFloat16);
    TORCH_CHECK(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(go.dtype() == at::kBFloat16);
    TORCH_CHECK(go.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alphas.dtype() == at::kFloat);
    TORCH_CHECK(alphas.numel() == 2);
    TORCH_CHECK(alphas.device().type() == at::DeviceType::CPU);
    TORCH_CHECK(beta > 0.0);
    TORCH_CHECK(eps > 0.0);

    at::Tensor gi = at::empty(x.sizes(), x.options());
    at::Tensor da_p = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(x.device()));
    at::Tensor da_n = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

    __nv_bfloat16* x_ptr = reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16* go_ptr = reinterpret_cast<__nv_bfloat16*>(go.data_ptr<at::BFloat16>());
    __nv_bfloat16* gi_ptr = reinterpret_cast<__nv_bfloat16*>(gi.data_ptr<at::BFloat16>());
    float* da_p_ptr = da_p.data_ptr<float>();
    float* da_n_ptr = da_n.data_ptr<float>();

    float* alphas_ptr = alphas.data_ptr<float>();
    float s_a_p = softplus_forward(alphas_ptr[0]);
    float s_a_n = softplus_forward(alphas_ptr[1]);
    float ds_a_p = softplus_backward(alphas_ptr[0]);
    float ds_a_n = softplus_backward(alphas_ptr[1]);

    int N = x.numel();
    int n_blocks = (N + BACKWARD_BLOCK_SIZE - 1) / BACKWARD_BLOCK_SIZE;

    xielu_backward_kernel<<<n_blocks, N_THREADS>>>(x_ptr, go_ptr, gi_ptr, da_p_ptr, da_n_ptr, s_a_p, s_a_n, ds_a_p, ds_a_n, static_cast<float>(beta), static_cast<float>(eps), N);

    return gi;
}

TORCH_LIBRARY(swissai, m) {
    m.def("xielu_forward(Tensor x, Tensor alphas, float beta, float eps) -> Tensor");
    m.impl("xielu_forward", &xielu_forward);

    m.def("xielu_backward(Tensor x, Tensor go, Tensor alphas, float beta, float eps) -> Tensor");
    m.impl("xielu_backward", &xielu_backward);
}

"""

load_inline(name="swissai", cpp_sources=[""], cuda_sources=[cuda_src], extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math", "--ptxas-options=-v"], verbose=True, is_python_module=False, no_implicit_headers=True)

if __name__ == "__main__":
    import time

    x = torch.randn(64 * 1024, 21504, dtype=torch.bfloat16, device="cuda")
    go = torch.randn_like(x, device="cuda")
    alphas = torch.tensor([0.8, 0.8], dtype=torch.float32, device="cpu")
    beta = 0.5
    eps = 1e-6

    _ = torch.ops.swissai.xielu_forward(x, alphas, beta, eps)
    _ = torch.ops.swissai.xielu_backward(x, go, alphas, beta, eps)
    # exit()

    for _ in range(10):
        _ = torch.ops.swissai.xielu_forward(x, alphas, beta, eps)
        torch.cuda.synchronize()

    n = 100
    start_time = time.perf_counter()
    for _ in range(n):
        _ = torch.ops.swissai.xielu_forward(x, alphas, beta, eps)
        torch.cuda.synchronize()
    print(f"xielu_froward: {(time.perf_counter() - start_time) * 1000 / n:.3f} ms")

    for _ in range(10):
        _ = torch.ops.swissai.xielu_backward(x, go, alphas, beta, eps)
        torch.cuda.synchronize()

    n = 100
    start_time = time.perf_counter()
    for _ in range(n):
        _ = torch.ops.swissai.xielu_backward(x, go, alphas, beta, eps)
        torch.cuda.synchronize()
    print(f"xielu_backward: {(time.perf_counter() - start_time) * 1000 / n:.3f} ms")
