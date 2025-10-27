import os
import torch
import numpy as np

from src.dataset import PorousDataset


def test_porous_dataset_train_and_reuse_scalers(tmp_path):
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

    ds_train = PorousDataset(
        X_dir=str(xdir),
        m_path=str(m_path),
        kappa_path=str(kappa_path),
        nums_data=2,
        m_scaler_name="minmax",
        kappa_scaler_name="log_zscore",
        train=True,
    )

    assert len(ds_train) == 2
    assert ds_train.m_scaler is not None
    assert ds_train.kappa_scaler is not None

    ds_val = PorousDataset(
        X_dir=str(xdir),
        m_path=str(m_path),
        kappa_path=str(kappa_path),
        nums_data=2,
        m_scaler_name="minmax",
        kappa_scaler_name="log_zscore",
        train=False,
        m_scaler=ds_train.m_scaler,
        kappa_scaler=ds_train.kappa_scaler,
    )

    assert ds_val.m_scaler is not None
    assert ds_val.kappa_scaler is not None

    m_np = np.asarray(m).reshape(-1, 1)
    m_scaled_train = ds_train.m_scaler.transform(m_np)
    m_scaled_val = ds_val.m_scaler.transform(m_np)
    assert np.allclose(m_scaled_train, m_scaled_val)

    kappa_np = np.asarray(kappa).reshape(-1, 1)
    kappa_scaled_train = ds_train.kappa_scaler.transform(kappa_np)
    kappa_scaled_val = ds_val.kappa_scaler.transform(kappa_np)
    assert np.allclose(kappa_scaled_train, kappa_scaled_val)