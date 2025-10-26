import pytest
import torch
from src.metrics import get_metrics


def test_r2_perfect_prediction():
    y = torch.tensor([1.0, 2.0, 3.0])
    y_pred = y.clone()
    r2 = get_metrics('r2', y, y_pred)
    assert pytest.approx(r2, rel=1e-6) == 1.0


def test_mse():
    y = torch.tensor([0.0, 0.0, 0.0])
    y_pred = torch.tensor([1.0, -1.0, 0.0])
    mse = get_metrics('mse', y, y_pred)
    assert pytest.approx(mse, rel=1e-6) == 2.0 / 3.0
