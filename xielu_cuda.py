import torch
from typing import Tuple
from torch.utils.cpp_extension import load_inline

cuda_src = """
#include <tuple>
#include <stdio.h>
#include <iostream>
#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/autograd.h>
#include <c10/cuda/CUDAException.h>

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
            y_local[j] = __float2bfloat16_rn(y);
        }

        *(uint4*)(y + (int)blockIdx.x * FORWARD_BLOCK_SIZE + i * FORWARD_CHUNK_SIZE + (int)threadIdx.x * 8) = *(uint4*)y_local;
    }
}

__global__ void xielu_backward_kernel(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ go, __nv_bfloat16* __restrict__ gi, float* __restrict__ galphas, float s_a_p, float s_a_n, float ds_a_p, float ds_a_n, float beta, float eps, int N) {
    __nv_bfloat16 x_local[8];
    __nv_bfloat16 go_local[8];
    __nv_bfloat16 gi_local[8];
    float galphas_local[2] = {0.0};

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

            galphas_local[0] += (x > 0.0f) ? go * ds_a_p * x * x : 0.0f;
            galphas_local[1] += (x <= 0.0f) ? go * ds_a_n * ((e - 1.0f) - x) : 0.0f;
        }

        *(uint4*)(gi + (int)blockIdx.x * BACKWARD_BLOCK_SIZE + i * BACKWARD_CHUNK_SIZE + (int)threadIdx.x * 8) = *(uint4*)gi_local;
    }

    for (int i = 16; i > 0; i /= 2) {
        galphas_local[0] += __shfl_down_sync(0xffffffff, galphas_local[0], i, 32);
        galphas_local[1] += __shfl_down_sync(0xffffffff, galphas_local[1], i, 32);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(galphas, galphas_local[0]);
        atomicAdd(galphas + 1, galphas_local[1]);
    }
}

float softplus_forward(float x) {
    return (x > 20.0) ? x : ((x < -20.0) ? 0.0 : log1pf(expf(x)));
}

float softplus_backward(float x) {
    return (x > 20.0) ? 1.0 : ((x < -20.0) ? 0.0 : 1.0 / (1.0 + expf(-x)));
}

at::Tensor xielu_forward(const at::Tensor &x, const at::Tensor &alphas, double beta, double eps) {
    TORCH_CHECK(x.dtype() == at::kBFloat16);
    TORCH_CHECK(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alphas.dtype() == at::kFloat);
    TORCH_CHECK(alphas.numel() == 2);
    TORCH_CHECK(alphas.device().type() == at::DeviceType::CPU);

    at::Tensor y = at::empty(x.sizes(), x.options());

    __nv_bfloat16* x_ptr = reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16* y_ptr = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());

    float* alphas_ptr = alphas.data_ptr<float>();
    float s_a_p = softplus_forward(alphas_ptr[0]);
    float s_a_n = softplus_forward(alphas_ptr[1]);

    int N = x.numel();
    int n_blocks = (N + FORWARD_BLOCK_SIZE - 1) / FORWARD_BLOCK_SIZE;
    
    xielu_forward_kernel<<<n_blocks, N_THREADS>>>(x_ptr, y_ptr, s_a_p, s_a_n, static_cast<float>(beta), static_cast<float>(eps), N);
    C10_CUDA_CHECK(cudaGetLastError());

    return y;
}

std::tuple<at::Tensor, at::Tensor> xielu_backward(const at::Tensor &x, const at::Tensor &go, const at::Tensor &alphas, double beta, double eps) {
    TORCH_CHECK(x.dtype() == at::kBFloat16);
    TORCH_CHECK(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(go.dtype() == at::kBFloat16);
    TORCH_CHECK(go.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alphas.dtype() == at::kFloat);
    TORCH_CHECK(alphas.numel() == 2);
    TORCH_CHECK(alphas.device().type() == at::DeviceType::CPU);

    at::Tensor gi = at::empty(x.sizes(), x.options());
    at::Tensor galphas = at::zeros({2}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

    __nv_bfloat16* x_ptr = reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16* go_ptr = reinterpret_cast<__nv_bfloat16*>(go.data_ptr<at::BFloat16>());
    __nv_bfloat16* gi_ptr = reinterpret_cast<__nv_bfloat16*>(gi.data_ptr<at::BFloat16>());
    float* galphas_ptr = galphas.data_ptr<float>();

    float* alphas_ptr = alphas.data_ptr<float>();
    float s_a_p = softplus_forward(alphas_ptr[0]);
    float s_a_n = softplus_forward(alphas_ptr[1]);
    float ds_a_p = softplus_backward(alphas_ptr[0]);
    float ds_a_n = softplus_backward(alphas_ptr[1]);

    int N = x.numel();
    int n_blocks = (N + BACKWARD_BLOCK_SIZE - 1) / BACKWARD_BLOCK_SIZE;

    xielu_backward_kernel<<<n_blocks, N_THREADS>>>(x_ptr, go_ptr, gi_ptr, galphas_ptr, s_a_p, s_a_n, ds_a_p, ds_a_n, static_cast<float>(beta), static_cast<float>(eps), N);
    C10_CUDA_CHECK(cudaGetLastError());

    at::Tensor galphas_cpu = galphas.to(at::DeviceType::CPU);
    return std::make_tuple(gi, galphas_cpu);
}

class XieLUFunction : public torch::autograd::Function<XieLUFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &x,
        const at::Tensor &alphas,
        double beta,
        double eps
    ) {
        ctx->save_for_backward({x, alphas});
        ctx->saved_data["beta"] = beta;
        ctx->saved_data["eps"] = eps;
        return xielu_forward(x, alphas, beta, eps);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        at::Tensor x = saved[0];
        at::Tensor alphas = saved[1];
        double beta = ctx->saved_data["beta"].toDouble();
        double eps = ctx->saved_data["eps"].toDouble();
        
        at::Tensor go = grad_outputs[0];
        if (!go.is_contiguous()) {
            go = go.contiguous();
        }

        if (!x.is_contiguous()) {
            x = x.contiguous();
        }

        if (alphas.device().type() != at::DeviceType::CPU) {
            alphas = alphas.to(at::DeviceType::CPU);
        }

        auto result = xielu_backward(x, go, alphas, beta, eps);
        at::Tensor gi = std::get<0>(result);
        at::Tensor galphas = std::get<1>(result);

        return {gi, galphas, at::Tensor(), at::Tensor()};
    }
};

at::Tensor xielu_autograd(const at::Tensor &x, const at::Tensor &alphas, double beta, double eps) {
    return XieLUFunction::apply(x, alphas, beta, eps);
}

TORCH_LIBRARY(swissai, m) {
    m.def("xielu_forward(Tensor x, Tensor alphas, float beta, float eps) -> Tensor");
    m.impl("xielu_forward", &xielu_forward);

    m.def("xielu_backward(Tensor x, Tensor go, Tensor alphas, float beta, float eps) -> (Tensor, Tensor)");
    m.impl("xielu_backward", &xielu_backward);

    m.def("xielu(Tensor x, Tensor alphas, float beta, float eps) -> Tensor");
    m.impl("xielu", TORCH_FN(xielu_autograd));
}

"""

load_inline(name="swissai", cpp_sources=[""], cuda_sources=[cuda_src], extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math", "--ptxas-options=-v"], verbose=True, is_python_module=False, no_implicit_headers=True)

def xielu(x: torch.Tensor, alphas: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    return torch.ops.swissai.xielu(x, alphas, beta, eps)

class xIELU(torch.nn.Module):
    def __init__(self, alpha_p: float = 0.8, alpha_n: float = 0.8, beta: float = 0.5, eps: float = -1e-6, device: torch.device = None, dtype: torch.dtype = None, **kwargs):
        super().__init__()

        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(abs(eps), dtype=torch.float32))

        with torch.no_grad():
            alphas_init = torch.log(torch.expm1(torch.tensor([alpha_p, alpha_n - beta], dtype=torch.float32)))
            self.alphas = torch.nn.Parameter(alphas_init.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.swissai.xielu(x, self.alphas, float(self.beta.item()), float(self.eps.item()))

    def _apply(self, fn):
        alphas_data = self.alphas.data.clone()
        super()._apply(fn)
        self.alphas.data = alphas_data.cpu()
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        alpha_p_key = prefix + "alpha_p"
        alpha_n_key = prefix + "alpha_n"
        alphas_key = prefix + "alphas"
        
        if alpha_p_key in state_dict and alpha_n_key in state_dict and alphas_key not in state_dict:
            alpha_p = state_dict[alpha_p_key].squeeze(0)
            alpha_n = state_dict[alpha_n_key].squeeze(0)
            
            alphas = torch.stack([alpha_p, alpha_n], dim=0).unsqueeze(0)
            state_dict[alphas_key] = alphas
            
            del state_dict[alpha_p_key]
            del state_dict[alpha_n_key]
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        if hasattr(self, "alphas"):
            self.alphas.data = self.alphas.data.cpu()

    

if __name__ == "__main__":
    import time

    x = torch.randn(64, 1024, 21504, dtype=torch.bfloat16, device="cuda")
    go = torch.randn_like(x, device="cuda")
    alphas = torch.tensor([0.8, 0.8], dtype=torch.float32, device="cpu")
    beta = 0.5
    eps = 1e-6

    # Warmup
    _ = torch.ops.swissai.xielu_forward(x, alphas, beta, eps)
    _ = torch.ops.swissai.xielu_backward(x, go, alphas, beta, eps)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()

    # Benchmark forward pass
    torch.cuda.nvtx.range_push("xielu_forward_warmup")
    for _ in range(10):
        _ = torch.ops.swissai.xielu_forward(x, alphas, beta, eps)
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("xielu_forward")
    n = 100
    start_time = time.perf_counter()
    for _ in range(n):
        _ = torch.ops.swissai.xielu_forward(x, alphas, beta, eps)
        torch.cuda.synchronize()
    print(f"xielu_froward: {(time.perf_counter() - start_time) * 1000 / n:.3f} ms")
    torch.cuda.nvtx.range_pop()

    # Benchmark backward pass
    torch.cuda.nvtx.range_push("xielu_backward_warmup")
    for _ in range(10):
        _, _ = torch.ops.swissai.xielu_backward(x, go, alphas, beta, eps)
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("xielu_backward")
    n = 100
    start_time = time.perf_counter()
    for _ in range(n):
        _, _ = torch.ops.swissai.xielu_backward(x, go, alphas, beta, eps)
        torch.cuda.synchronize()
    print(f"xielu_backward: {(time.perf_counter() - start_time) * 1000 / n:.3f} ms")
    torch.cuda.nvtx.range_pop()

    # Create tensors with requires_grad for autograd test
    x_grad = torch.randn(64, 1024, 21504, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    alphas_grad = torch.tensor([0.8, 0.8], dtype=torch.float32, device="cpu", requires_grad=True)
    
    torch.cuda.nvtx.range_push("xielu_autograd_warmup")
    for _ in range(10):
        x_grad.grad = None
        alphas_grad.grad = None
        y = xielu(x_grad, alphas_grad, beta, eps)
        y.backward(go)
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("xielu_autograd")
    n = 100
    start_time = time.perf_counter()
    for _ in range(n):
        x_grad.grad = None
        alphas_grad.grad = None
        y = xielu(x_grad, alphas_grad, beta, eps)
        y.backward(go)
        torch.cuda.synchronize()
    print(f"end to end: {(time.perf_counter() - start_time) * 1000 / n:.3f} ms")
    torch.cuda.nvtx.range_pop()
