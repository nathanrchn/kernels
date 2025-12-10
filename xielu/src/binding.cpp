#include "xielu.h"
#include <torch/extension.h>

TORCH_LIBRARY(xielu, m) {
    m.def("forward(Tensor x, Tensor alpha_p, Tensor alpha_n, float beta, float eps) -> Tensor", &xielu_forward);
    m.def("backward(Tensor x, Tensor go, Tensor alpha_p, Tensor alpha_n, float beta, float eps) -> (Tensor, Tensor, Tensor)", &xielu_backward);
}
