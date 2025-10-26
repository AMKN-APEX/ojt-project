import torch
import pytest
import os
from src.dataset import PorousDataset
from src.normalizer import apply_scaler


def test_porous_dataset_returns_expected_types(tmp_path):
    xdir = tmp_path / "xdir"
    xdir.mkdir()
    x0 = torch.randn(3, 3)
    x1 = torch.randn(3, 3)
    torch.save(x0, os.path.join(xdir, "0.pt"))
    torch.save(x1, os.path.join(xdir, "1.pt"))

    m = torch.tensor([0.0, 1.0])
    kappa = torch.tensor([1.0, 4.0])
    m_path = tmp_path / "m.pt"
    kappa_path = tmp_path / "kappa.pt"
    torch.save(m, str(m_path))
    torch.save(kappa, str(kappa_path))

    ds = PorousDataset(
        X_dir=str(xdir),
        m_path=str(m_path),
        kappa_path=str(kappa_path),
        nums_data=2,
        m_scaler_name="minmax",
        kappa_scaler_name="log_zscore",
    )

    assert len(ds) == 2
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, list) and len(y) == 2
    assert isinstance(y[0], torch.Tensor)
    assert y[0].dtype == torch.float32
    assert isinstance(y[1], torch.Tensor)
    assert y[1].dtype == torch.float32

    all_m_scaled = ds.m_scaled
    assert torch.is_tensor(all_m_scaled)
    assert all_m_scaled.shape[0] == 2

