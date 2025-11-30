#include "xielu.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>

constexpr int N_THREADS = 256;
constexpr int VECTOR_SIZE = 8;
constexpr int MAX_BLOCKS = 2048;
constexpr int MIN_ALIGNMENT = 128;

__global__ void xielu_forward_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const float s_a_p,
    const float alpha_n,
    const float neg_s_a_n,
    const float beta,
    const float eps,
    const int64_t num_vectors
) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t vec_idx = tid; vec_idx < num_vectors; vec_idx += stride) {
        const int64_t offset = vec_idx * VECTOR_SIZE;
        
        uint4 x_data = __ldg(reinterpret_cast<const uint4*>(x + offset));
        const __nv_bfloat16* x_local = reinterpret_cast<const __nv_bfloat16*>(&x_data);
        
        uint4 y_data;
        __nv_bfloat16* y_local = reinterpret_cast<__nv_bfloat16*>(&y_data);
        
        #pragma unroll
        for (int j = 0; j < VECTOR_SIZE; j++) {
            float xf = __bfloat162float(x_local[j]);
            float yf;
            if (xf > 0.0f) {
                yf = xf * __fmaf_rn(s_a_p, xf, beta);
            } else {
                float e = __expf(fminf(xf, eps)) - 1.0f;
                yf = __fmaf_rn(alpha_n, e, neg_s_a_n * xf);
            }
            y_local[j] = __float2bfloat16_rn(yf);
        }
        
        *reinterpret_cast<uint4*>(y + offset) = y_data;
    }
}

__global__ void xielu_backward_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ go,
    __nv_bfloat16* __restrict__ gi,
    float* __restrict__ galpha_p,
    float* __restrict__ galpha_n,
    const float two_s_a_p,
    const float alpha_n,
    const float ds_a_p,
    const float ds_a_n,
    const float beta,
    const float eps,
    const int64_t num_vectors
) {
    float galpha_p_local = 0.0f;
    float galpha_n_local = 0.0f;
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t vec_idx = tid; vec_idx < num_vectors; vec_idx += stride) {
        const int64_t offset = vec_idx * VECTOR_SIZE;
        
        uint4 x_data = __ldg(reinterpret_cast<const uint4*>(x + offset));
        uint4 go_data = __ldg(reinterpret_cast<const uint4*>(go + offset));
        const __nv_bfloat16* x_local = reinterpret_cast<const __nv_bfloat16*>(&x_data);
        const __nv_bfloat16* go_local = reinterpret_cast<const __nv_bfloat16*>(&go_data);
        
        uint4 gi_data;
        __nv_bfloat16* gi_local = reinterpret_cast<__nv_bfloat16*>(&gi_data);
        
        #pragma unroll
        for (int j = 0; j < VECTOR_SIZE; j++) {
            float xf = __bfloat162float(x_local[j]);
            float gof = __bfloat162float(go_local[j]);
            
            float dx, contrib_p = 0.0f, contrib_n = 0.0f;
            
            if (xf > 0.0f) {
                dx = gof * __fmaf_rn(two_s_a_p, xf, beta);
                contrib_p = gof * ds_a_p * xf * xf;
            } else {
                float e = __expf(fminf(xf, eps));
                float below_eps = (xf <= eps) ? 1.0f : 0.0f;
                dx = gof * __fmaf_rn(alpha_n, e * below_eps - 1.0f, beta);
                contrib_n = gof * ds_a_n * ((e - 1.0f) - xf);
            }
            
            gi_local[j] = __float2bfloat16_rn(dx);
            galpha_p_local += contrib_p;
            galpha_n_local += contrib_n;
        }
        
        *reinterpret_cast<uint4*>(gi + offset) = gi_data;
    }

    // Warp reduction
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        galpha_p_local += __shfl_down_sync(0xffffffff, galpha_p_local, i, 32);
        galpha_n_local += __shfl_down_sync(0xffffffff, galpha_n_local, i, 32);
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(galpha_p, galpha_p_local);
        atomicAdd(galpha_n, galpha_n_local);
    }
}

inline float softplus_forward(float x) {
    return (x > 20.0f) ? x : ((x < -20.0f) ? 0.0f : log1pf(expf(x)));
}

inline float softplus_backward(float x) {
    return (x > 20.0f) ? 1.0f : ((x < -20.0f) ? 0.0f : 1.0f / (1.0f + expf(-x)));
}

inline int compute_num_blocks(int64_t num_vectors) {
    int n_blocks = static_cast<int>((num_vectors + N_THREADS - 1) / N_THREADS);
    return std::max(1, std::min(n_blocks, MAX_BLOCKS));
}

at::Tensor xielu_forward(const at::Tensor &x, const at::Tensor &alpha_p, const at::Tensor &alpha_n, double beta, double eps) {
    TORCH_CHECK(x.dtype() == at::kBFloat16);
    TORCH_CHECK(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alpha_p.dtype() == at::kBFloat16);
    TORCH_CHECK(alpha_p.numel() == 1);
    TORCH_CHECK(alpha_p.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alpha_n.dtype() == at::kBFloat16);
    TORCH_CHECK(alpha_n.numel() == 1);
    TORCH_CHECK(alpha_n.device().type() == at::DeviceType::CUDA);

    int64_t N = x.numel();
    TORCH_CHECK(N % MIN_ALIGNMENT == 0, "Input size must be divisible by ", MIN_ALIGNMENT);

    at::Tensor y = at::empty(x.sizes(), x.options());

    const __nv_bfloat16* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16* y_ptr = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());

    __nv_bfloat16 alpha_p_host;
    __nv_bfloat16 alpha_n_host;
    __nv_bfloat16* alpha_p_cuda_ptr = reinterpret_cast<__nv_bfloat16*>(alpha_p.data_ptr<at::BFloat16>());
    __nv_bfloat16* alpha_n_cuda_ptr = reinterpret_cast<__nv_bfloat16*>(alpha_n.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaMemcpy(&alpha_p_host, alpha_p_cuda_ptr, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(&alpha_n_host, alpha_n_cuda_ptr, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float s_a_p = softplus_forward(__bfloat162float(alpha_p_host));
    float s_a_n = softplus_forward(__bfloat162float(alpha_n_host));
    float alpha_n_val = static_cast<float>(beta) + s_a_n;
    float neg_s_a_n = -s_a_n;

    int64_t num_vectors = N / VECTOR_SIZE;
    int n_blocks = compute_num_blocks(num_vectors);
    
    xielu_forward_kernel<<<n_blocks, N_THREADS>>>(
        x_ptr, y_ptr, s_a_p, alpha_n_val, neg_s_a_n,
        static_cast<float>(beta), static_cast<float>(eps), num_vectors
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> xielu_backward(const at::Tensor &x, const at::Tensor &go, const at::Tensor &alpha_p, const at::Tensor &alpha_n, double beta, double eps) {
    TORCH_CHECK(x.dtype() == at::kBFloat16);
    TORCH_CHECK(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(go.dtype() == at::kBFloat16);
    TORCH_CHECK(go.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alpha_p.dtype() == at::kBFloat16);
    TORCH_CHECK(alpha_p.numel() == 1);
    TORCH_CHECK(alpha_p.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(alpha_n.dtype() == at::kBFloat16);
    TORCH_CHECK(alpha_n.numel() == 1);
    TORCH_CHECK(alpha_n.device().type() == at::DeviceType::CUDA);

    int64_t N = x.numel();
    TORCH_CHECK(N % MIN_ALIGNMENT == 0, "Input size must be divisible by ", MIN_ALIGNMENT);

    at::Tensor gi = at::empty(x.sizes(), x.options());
    at::Tensor galpha_p_tensor = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(x.device()));
    at::Tensor galpha_n_tensor = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(x.device()));

    const __nv_bfloat16* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    const __nv_bfloat16* go_ptr = reinterpret_cast<const __nv_bfloat16*>(go.data_ptr<at::BFloat16>());
    __nv_bfloat16* gi_ptr = reinterpret_cast<__nv_bfloat16*>(gi.data_ptr<at::BFloat16>());
    float* galpha_p_ptr = galpha_p_tensor.data_ptr<float>();
    float* galpha_n_ptr = galpha_n_tensor.data_ptr<float>();

    __nv_bfloat16 alpha_p_host;
    __nv_bfloat16 alpha_n_host;
    __nv_bfloat16* alpha_p_cuda_ptr = reinterpret_cast<__nv_bfloat16*>(alpha_p.data_ptr<at::BFloat16>());
    __nv_bfloat16* alpha_n_cuda_ptr = reinterpret_cast<__nv_bfloat16*>(alpha_n.data_ptr<at::BFloat16>());
    C10_CUDA_CHECK(cudaMemcpy(&alpha_p_host, alpha_p_cuda_ptr, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(&alpha_n_host, alpha_n_cuda_ptr, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float s_a_p = softplus_forward(__bfloat162float(alpha_p_host));
    float s_a_n = softplus_forward(__bfloat162float(alpha_n_host));
    float ds_a_p = softplus_backward(__bfloat162float(alpha_p_host));
    float ds_a_n = softplus_backward(__bfloat162float(alpha_n_host));
    
    float two_s_a_p = 2.0f * s_a_p;
    float alpha_n_val = static_cast<float>(beta) + s_a_n;

    int64_t num_vectors = N / VECTOR_SIZE;
    int n_blocks = compute_num_blocks(num_vectors);

    xielu_backward_kernel<<<n_blocks, N_THREADS>>>(
        x_ptr, go_ptr, gi_ptr, galpha_p_ptr, galpha_n_ptr,
        two_s_a_p, alpha_n_val, ds_a_p, ds_a_n,
        static_cast<float>(beta), static_cast<float>(eps), num_vectors
    );
    C10_CUDA_CHECK(cudaGetLastError());
    
    return std::make_tuple(gi, galpha_p_tensor, galpha_n_tensor);
}
