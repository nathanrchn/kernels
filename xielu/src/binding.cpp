#include "xielu.h"
#include <torch/extension.h>

TORCH_LIBRARY(xielu, m) {
    m.def("forward(Tensor x, Tensor alpha_p, Tensor alpha_n, float beta, float eps) -> Tensor");
    m.def("backward(Tensor x, Tensor go, Tensor alpha_p, Tensor alpha_n, float beta, float eps) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(xielu, CUDA, m) {
    m.impl("forward", &xielu_forward);
    m.impl("backward", &xielu_backward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &xielu_forward);
    m.def("backward", &xielu_backward);
}
