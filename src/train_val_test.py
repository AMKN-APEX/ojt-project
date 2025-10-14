import torch
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch import nn, optim


@dataclass
class TrainValTest:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer
    device: torch.device
    num_epochs: int = 20


    def __post_init__(self):
        self.model.to(self.device)
        print(self.device)


    def run_epoch(self, loader: DataLoader, train: bool = True) -> list:
        loss1_total = 0.0
        loss2_total = 0.0
        self.model.train(train)
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device, dtype=torch.float32)
                y1 = y[0].to(self.device, dtype=torch.float32)
                y2 = y[1].to(self.device, dtype=torch.float32)

                outputs = self.model(x)
                outputs1 = outputs[:, 0].view(-1, 1)
                outputs2 = outputs[:, 1].view(-1, 1)

                loss1 = self.criterion(outputs1, y1)
                loss2 = self.criterion(outputs2, y2)
                loss = loss1 + loss2

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                loss1_total += loss1.item() * x.size(0)
                loss2_total += loss2.item() * x.size(0)
        return loss1_total / len(loader.dataset), loss2_total / len(loader.dataset)  # type: ignore


    def train_val(self):
        for epoch in range(self.num_epochs):
            train_loss_m, train_loss_k = self.run_epoch(self.train_loader, train=True)
            val_loss_m, val_loss_k = self.run_epoch(self.val_loader, train=False)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss m: {train_loss_m:.4f}, Train Loss kappa: {train_loss_k:.4f}, Val Loss m: {val_loss_m:.4f}, Val Loss kappa: {val_loss_k:.4f}")


    def test(self):
        test_loss_m, test_loss_k = self.run_epoch(self.test_loader, train=False)
        print(f"Test Loss m: {test_loss_m:.4f}, Test Loss kappa: {test_loss_k:.4f}")