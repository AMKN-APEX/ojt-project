import torch
import pytest
from torch.utils.data import Dataset, DataLoader
from torch import nn
from src.train_val_test import TrainValTest


class SimpleDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = [torch.tensor(p, dtype=torch.float32) for p in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x = self.pairs[idx]
        m = x[0].view(1) if x.ndim == 1 else x[0:1]
        k = x[1].view(1) if x.ndim == 1 else x[1:2]
        return x.unsqueeze(0).squeeze(0), [m, k]


class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


def test_run_epoch_perfect_prediction():
    pairs = [(1.0, 2.0), (3.0, 4.0)]
    ds = SimpleDataset(pairs)
    loader = DataLoader(ds, batch_size=2)

    model = IdentityModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    runner = TrainValTest(
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cpu'),
        metrics_name='r2',
        num_epochs=1,
        scheduler=None,
    )

    loss_m, loss_k, metric_m, metric_k = runner.run_epoch(loader, train=False)
    assert loss_m < 1e-6
    assert loss_k < 1e-6
    assert pytest.approx(metric_m, rel=1e-6) == 1.0
    assert pytest.approx(metric_k, rel=1e-6) == 1.0
