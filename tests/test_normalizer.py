import pytest
import torch
from src.normalizer import apply_scaler


def test_minmax_scaler_returns_tensor_and_shape():
    x = torch.tensor([0.0, 1.0, 2.0])
    scaled = apply_scaler('minmax', x)
    assert isinstance(scaled, torch.Tensor)
    assert scaled.dtype == torch.float32
    assert scaled.ndim == 2 and scaled.shape[1] == 1
    vals = scaled.squeeze().tolist()
    assert pytest.approx(vals[0], rel=1e-6) == 0.0
    assert pytest.approx(vals[1], rel=1e-6) == 0.5
    assert pytest.approx(vals[2], rel=1e-6) == 1.0


def test_log_zscore_outputs_zero_mean():
    x = torch.tensor([1.0, 2.0, 4.0, 8.0])
    scaled = apply_scaler('log_zscore', x)
    assert isinstance(scaled, torch.Tensor)
    mean = scaled.mean().item()
    assert abs(mean) < 1e-6
