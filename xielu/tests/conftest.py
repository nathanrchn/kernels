import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Get CUDA device"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")
