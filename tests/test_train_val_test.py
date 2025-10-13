import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from src.model import CNN
from src.train_val_test import TrainValTest


def _make_loader(num_samples=8, batch_size=4):
    # 小さなダミーデータセットを作成
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randn(num_samples, 1)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size)


def test_trainval_run_epoch():
    """TrainValTest の run_epoch が例外を出さずに動作することを確認する。"""
    train_loader = _make_loader()
    val_loader = _make_loader()
    test_loader = _make_loader()

    device = torch.device("cpu")
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    runner = TrainValTest(train_loader, val_loader, test_loader, model, criterion, optimizer, device, num_epochs=1)

    runner.train_val()
    runner.test()
