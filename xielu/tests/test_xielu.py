import pytest
import torch
import xielu as xielu_lib


def xielu_torch_reference(
    x: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    beta: float,
    eps: torch.Tensor
) -> torch.Tensor:
    """Reference implementation in PyTorch"""
    x = x.to(torch.float32)
    return torch.where(
        x > 0,
        alpha_p * x * x + beta * x,
        alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x
    ).to(torch.bfloat16)


class TestXieLUForward:
    """Test XIELU forward pass with various input distributions"""

    @pytest.mark.parametrize("shape", [
        (128,),
        (256, 512),
        (32, 64, 128),
        (16, 32, 64, 128),
    ])
    @pytest.mark.parametrize("alpha_p,alpha_n,beta,eps", [
        (0.8, 0.8, 0.5, -1e-6),
        (166.0, 40.75, 0.5, -1e-3),
        (1.0, 1.0, 0.0, -1e-4),
        (0.5, 1.5, 0.25, -1e-5),
    ])
    def test_uniform_random(self, device, shape, alpha_p, alpha_n, beta, eps):
        """Test with uniformly distributed random inputs"""
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
        alpha_p_t = torch.tensor(alpha_p, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n_t = torch.tensor(alpha_n, dtype=torch.bfloat16, device=device).unsqueeze(0)
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p_t, alpha_n_t, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p_t, alpha_n_t, beta, eps_t)

        # bfloat16 has ~3 decimal digits of precision
        # Use relative tolerance of 1e-2 (1%) and absolute tolerance of 1e-3
        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3), \
            f"Max error: {torch.max(torch.abs(y_cuda - y_ref))}"

    @pytest.mark.parametrize("range_min,range_max", [
        (-10.0, 10.0),
        (-1.0, 1.0),
        (-100.0, 100.0),
        (-0.1, 0.1),
    ])
    def test_uniform_distribution_ranges(self, device, range_min, range_max):
        """Test with uniform distributions in different ranges"""
        shape = (256, 1024)
        x = torch.rand(shape, dtype=torch.bfloat16, device=device) * (range_max - range_min) + range_min

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("mean,std", [
        (0.0, 1.0),
        (0.0, 5.0),
        (2.0, 1.0),
        (-2.0, 0.5),
    ])
    def test_normal_distribution(self, device, mean, std):
        """Test with normally distributed inputs"""
        shape = (512, 512)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device) * std + mean

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_positive_only(self, device):
        """Test with only positive values"""
        shape = (256, 512)
        x = torch.abs(torch.randn(shape, dtype=torch.bfloat16, device=device))

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_negative_only(self, device):
        """Test with only negative values"""
        shape = (256, 512)
        x = -torch.abs(torch.randn(shape, dtype=torch.bfloat16, device=device))

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_all_zeros(self, device):
        """Test with all zero values"""
        shape = (128, 256)
        x = torch.zeros(shape, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_constant_positive(self, device):
        """Test with constant positive values"""
        shape = (256, 512)
        x = torch.full(shape, 5.0, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_constant_negative(self, device):
        """Test with constant negative values"""
        shape = (256, 512)
        x = torch.full(shape, -1.0, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_mixed_distribution(self, device):
        """Test with a mixture of different distributions"""
        shape = (256, 512)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)

        # Create regions with different characteristics
        mask_pos = x > 0.5
        mask_neg = x < -0.5
        mask_zero = (~mask_pos) & (~mask_neg)

        x[mask_pos] = torch.abs(x[mask_pos]) * 10  # Large positive
        x[mask_neg] = -torch.abs(x[mask_neg]) * 5   # Moderate negative
        x[mask_zero] = x[mask_zero] * 0.1           # Small values near zero

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_near_eps_boundary(self, device):
        """Test values near the eps boundary"""
        eps = -1e-3
        shape = (256, 512)

        # Create values around the eps boundary
        x = torch.linspace(eps * 2, eps / 2, shape[0] * shape[1], dtype=torch.bfloat16, device=device).reshape(shape)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_large_tensor(self, device):
        """Test with large tensor size"""
        shape = (64, 1024, 21504)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(166.0, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(40.75, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-3
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        # For large tensors, we can be slightly more lenient
        assert torch.allclose(y_cuda, y_ref, rtol=2e-2, atol=2e-3)


class TestXieLUBackward:
    """Test XIELU backward pass with various input distributions"""

    @pytest.mark.parametrize("shape", [
        (128,),
        (256, 512),
        (32, 64, 128),
    ])
    def test_gradients_uniform_random(self, device, shape):
        """Test gradients with uniformly distributed random inputs"""
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
        go = torch.randn(shape, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6

        # Compute gradients with CUDA kernel
        gi_cuda, galpha_p_cuda, galpha_n_cuda = xielu_lib.xielu_backward(
            x, go, alpha_p, alpha_n, beta, eps
        )

        # Compute gradients with PyTorch
        x_ref = x.clone().requires_grad_(True)
        alpha_p_ref = alpha_p.clone().requires_grad_(True)
        alpha_n_ref = alpha_n.clone().requires_grad_(True)
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_ref = xielu_torch_reference(x_ref, alpha_p_ref, alpha_n_ref, beta, eps_t)
        y_ref.backward(go)

        # Compare gradients
        assert torch.allclose(gi_cuda, x_ref.grad, rtol=1e-2, atol=1e-3), \
            f"x gradient max error: {torch.max(torch.abs(gi_cuda - x_ref.grad))}"

        # For scalar gradients, use slightly higher tolerance
        assert torch.allclose(galpha_p_cuda, alpha_p_ref.grad, rtol=5e-2, atol=1e-2), \
            f"alpha_p gradient error: {torch.abs(galpha_p_cuda - alpha_p_ref.grad)}"

        assert torch.allclose(galpha_n_cuda, alpha_n_ref.grad, rtol=5e-2, atol=1e-2), \
            f"alpha_n gradient error: {torch.abs(galpha_n_cuda - alpha_n_ref.grad)}"

    def test_gradients_positive_only(self, device):
        """Test gradients with only positive values"""
        shape = (256, 512)
        x = torch.abs(torch.randn(shape, dtype=torch.bfloat16, device=device))
        go = torch.randn(shape, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6

        gi_cuda, galpha_p_cuda, galpha_n_cuda = xielu_lib.xielu_backward(
            x, go, alpha_p, alpha_n, beta, eps
        )

        x_ref = x.clone().requires_grad_(True)
        alpha_p_ref = alpha_p.clone().requires_grad_(True)
        alpha_n_ref = alpha_n.clone().requires_grad_(True)
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_ref = xielu_torch_reference(x_ref, alpha_p_ref, alpha_n_ref, beta, eps_t)
        y_ref.backward(go)

        assert torch.allclose(gi_cuda, x_ref.grad, rtol=1e-2, atol=1e-3)
        assert torch.allclose(galpha_p_cuda, alpha_p_ref.grad, rtol=5e-2, atol=1e-2)
        # For positive-only inputs, alpha_n gradient should be very small
        assert torch.allclose(galpha_n_cuda, alpha_n_ref.grad, rtol=5e-2, atol=1e-2)

    def test_gradients_negative_only(self, device):
        """Test gradients with only negative values"""
        shape = (256, 512)
        x = -torch.abs(torch.randn(shape, dtype=torch.bfloat16, device=device))
        go = torch.randn(shape, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6

        gi_cuda, galpha_p_cuda, galpha_n_cuda = xielu_lib.xielu_backward(
            x, go, alpha_p, alpha_n, beta, eps
        )

        x_ref = x.clone().requires_grad_(True)
        alpha_p_ref = alpha_p.clone().requires_grad_(True)
        alpha_n_ref = alpha_n.clone().requires_grad_(True)
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_ref = xielu_torch_reference(x_ref, alpha_p_ref, alpha_n_ref, beta, eps_t)
        y_ref.backward(go)

        assert torch.allclose(gi_cuda, x_ref.grad, rtol=1e-2, atol=1e-3)
        # For negative-only inputs, alpha_p gradient should be very small
        assert torch.allclose(galpha_p_cuda, alpha_p_ref.grad, rtol=5e-2, atol=1e-2)
        assert torch.allclose(galpha_n_cuda, alpha_n_ref.grad, rtol=5e-2, atol=1e-2)


class TestXieLUAutograd:
    """Test XIELU autograd integration"""

    def test_autograd_integration(self, device):
        """Test autograd integration with backward pass"""
        shape = (256, 512)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device, requires_grad=True)
        go = torch.randn(shape, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device, requires_grad=True).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device, requires_grad=True).unsqueeze(0)
        beta = 0.5
        eps = -1e-6

        # Test autograd version
        y = xielu_lib.xielu(x, alpha_p, alpha_n, beta, eps)
        y.backward(go)

        gi_auto = x.grad.clone()
        galpha_p_auto = alpha_p.grad.clone()
        galpha_n_auto = alpha_n.grad.clone()

        # Test reference version
        x_ref = x.detach().clone().requires_grad_(True)
        alpha_p_ref = alpha_p.detach().clone().requires_grad_(True)
        alpha_n_ref = alpha_n.detach().clone().requires_grad_(True)
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_ref = xielu_torch_reference(x_ref, alpha_p_ref, alpha_n_ref, beta, eps_t)
        y_ref.backward(go)

        # Compare outputs
        assert torch.allclose(y, y_ref, rtol=1e-2, atol=1e-3)

        # Compare gradients
        assert torch.allclose(gi_auto, x_ref.grad, rtol=1e-2, atol=1e-3)
        assert torch.allclose(galpha_p_auto, alpha_p_ref.grad, rtol=5e-2, atol=1e-2)
        assert torch.allclose(galpha_n_auto, alpha_n_ref.grad, rtol=5e-2, atol=1e-2)

    @pytest.mark.parametrize("alpha_p,alpha_n,beta,eps", [
        (0.8, 0.8, 0.5, -1e-6),
        (166.0, 40.75, 0.5, -1e-3),
        (1.0, 1.0, 0.0, -1e-4),
    ])
    def test_autograd_different_params(self, device, alpha_p, alpha_n, beta, eps):
        """Test autograd with different parameter values"""
        shape = (128, 256)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device, requires_grad=True)
        go = torch.randn(shape, dtype=torch.bfloat16, device=device)

        alpha_p_t = torch.tensor(alpha_p, dtype=torch.bfloat16, device=device, requires_grad=True).unsqueeze(0)
        alpha_n_t = torch.tensor(alpha_n, dtype=torch.bfloat16, device=device, requires_grad=True).unsqueeze(0)

        y = xielu_lib.xielu(x, alpha_p_t, alpha_n_t, beta, eps)
        y.backward(go)

        assert x.grad is not None
        assert alpha_p_t.grad is not None
        assert alpha_n_t.grad is not None

        # Check gradients are not NaN or Inf
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(alpha_p_t.grad).any()
        assert not torch.isnan(alpha_n_t.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isinf(alpha_p_t.grad).any()
        assert not torch.isinf(alpha_n_t.grad).any()


class TestXieLUEdgeCases:
    """Test XIELU edge cases"""

    def test_very_small_values(self, device):
        """Test with very small input values"""
        shape = (256, 512)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device) * 1e-4

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        assert torch.allclose(y_cuda, y_ref, rtol=1e-2, atol=1e-3)

    def test_very_large_values(self, device):
        """Test with very large input values"""
        shape = (256, 512)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device) * 100

        alpha_p = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(0.8, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        # For very large values, allow higher tolerance
        assert torch.allclose(y_cuda, y_ref, rtol=5e-2, atol=1e-2)

    def test_extreme_alpha_values(self, device):
        """Test with extreme alpha values"""
        shape = (256, 512)
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)

        alpha_p = torch.tensor(200.0, dtype=torch.bfloat16, device=device).unsqueeze(0)
        alpha_n = torch.tensor(100.0, dtype=torch.bfloat16, device=device).unsqueeze(0)
        beta = 0.5
        eps = -1e-6
        eps_t = torch.tensor(eps, dtype=torch.float32, device=device)

        y_cuda = xielu_lib.xielu_forward(x, alpha_p, alpha_n, beta, eps)
        y_ref = xielu_torch_reference(x, alpha_p, alpha_n, beta, eps_t)

        # With extreme values, use higher tolerance
        assert torch.allclose(y_cuda, y_ref, rtol=5e-2, atol=1e-2)
