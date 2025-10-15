import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from src.model import CNN
from src.train_val_test import TrainValTest


class _PairDataset(Dataset):
    """
    簡易データセット: 各サンプルは (x, (y1, y2)) を返す。
    DataLoader のデフォルト collate によって、
    バッチ時には y が (y1_batch, y2_batch) のタプルになる。
    """

    def __init__(self, num_samples=8):
        self.x = torch.randn(num_samples, 3, 32, 32)
        self.y1 = torch.randn(num_samples, 1)
        self.y2 = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], (self.y1[idx], self.y2[idx])


def _make_loader(num_samples=8, batch_size=4):
    ds = _PairDataset(num_samples=num_samples)
    return DataLoader(ds, batch_size=batch_size)


def test_run_epoch_and_trainval():
    """run_epoch が 2 つの float を返し、train_val/test が例外を出さずに動くことを確認する。"""
    train_loader = _make_loader()
    val_loader = _make_loader()
    test_loader = _make_loader()

    device = torch.device("cpu")
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    runner = TrainValTest(train_loader, val_loader, test_loader, model, criterion, optimizer, device, num_epochs=1)

    # run_epoch を直接呼んで戻り値の型を検査
    train_loss_m, train_loss_k = runner.run_epoch(train_loader, train=True)
    assert isinstance(train_loss_m, float)
    assert isinstance(train_loss_k, float)

    # train_val と test を呼んで例外が出ないことを確認
    runner.train_val()
    runner.test()
