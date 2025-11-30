#include "xielu.h"
#include <torch/extension.h>
#include <torch/autograd.h>

class XieLUFunction : public torch::autograd::Function<XieLUFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &x,
        const at::Tensor &alpha_p,
        const at::Tensor &alpha_n,
        double beta,
        double eps
    ) {
        ctx->save_for_backward({x, alpha_p, alpha_n});
        ctx->saved_data["beta"] = beta;
        ctx->saved_data["eps"] = eps;
        return xielu_forward(x, alpha_p, alpha_n, beta, eps);
    }
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        at::Tensor x = saved[0];
        at::Tensor alpha_p = saved[1];
        at::Tensor alpha_n = saved[2];
        double beta = ctx->saved_data["beta"].toDouble();
        double eps = ctx->saved_data["eps"].toDouble();
        
        at::Tensor go = grad_outputs[0];
        if (!go.is_contiguous()) {
            go = go.contiguous();
        }

        if (!x.is_contiguous()) {
            x = x.contiguous();
        }

        auto result = xielu_backward(x, go, alpha_p, alpha_n, beta, eps);
        at::Tensor gi = std::get<0>(result);
        at::Tensor galpha_p = std::get<1>(result);
        at::Tensor galpha_n = std::get<2>(result);

        return {gi, galpha_p, galpha_n, at::Tensor(), at::Tensor()};
    }
};

at::Tensor xielu_autograd(const at::Tensor &x, const at::Tensor &alpha_p, const at::Tensor &alpha_n, double beta, double eps) {
    return XieLUFunction::apply(x, alpha_p, alpha_n, beta, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xielu", &xielu_autograd);
    m.def("xielu_forward", &xielu_forward);
    m.def("xielu_backward", &xielu_backward);
}
