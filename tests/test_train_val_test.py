import os
import torch
from torch.utils.data import DataLoader
from torch import nn

from src.dataset import PorousDataset
from src.train_val_test import TrainValTest
from src.criterion import get_criterion
from src.optimizer import get_optimizer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(9, 2)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


def make_tmp_dataset(tmp_path):
    xdir = tmp_path / "xdir"
    xdir.mkdir()
    for i in range(4):
        t = torch.randn(3, 3)
        torch.save(t, os.path.join(xdir, f"{i}.pt"))

    m = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kappa = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m_path = tmp_path / "m.pt"
    kappa_path = tmp_path / "kappa.pt"
    torch.save(m, str(m_path))
    torch.save(kappa, str(kappa_path))

    return str(xdir), str(m_path), str(kappa_path)


def test_train_val_train_once(tmp_path):
    X_dir, m_path, kappa_path = make_tmp_dataset(tmp_path)

    train_ds = PorousDataset(
        X_dir=X_dir,
        m_path=m_path,
        kappa_path=kappa_path,
        nums_data=4,
        m_scaler_name="zscore",
        kappa_scaler_name="minmax",
        train=True,
    )

    val_ds = PorousDataset(
        X_dir=X_dir,
        m_path=m_path,
        kappa_path=kappa_path,
        nums_data=4,
        m_scaler_name="zscore",
        kappa_scaler_name="minmax",
        train=False,
        m_scaler=train_ds.m_scaler,
        kappa_scaler=train_ds.kappa_scaler,
    )

    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    test_loader = DataLoader(val_ds, batch_size=2)

    model = DummyModel()
    criterion = get_criterion('mse')
    optimizer = get_optimizer('sgd', model.parameters(), 0.01)

    runner = TrainValTest(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cpu'),
        metrics_name='mae',
        num_epochs=1,
        scheduler=None,
        m_scaler_name='zscore',
        kappa_scaler_name='minmax',
        m_scaler=train_ds.m_scaler,
        kappa_scaler=train_ds.kappa_scaler,
    )

    runner.train_val()
    runner.test()
