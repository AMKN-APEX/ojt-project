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

    def run_epoch(self, loader: DataLoader, train: bool = True) -> float:
        loss_total = 0.0
        self.model.train(train)
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32).view(-1,1)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                loss_total += loss.item() * x.size(0)
        return loss_total / len(loader.dataset)  # type: ignore

    def train_val(self):
        for epoch in range(self.num_epochs):
            train_loss = self.run_epoch(self.train_loader, train=True)
            val_loss = self.run_epoch(self.val_loader, train=False)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def test(self):
        test_loss = self.run_epoch(self.test_loader, train=False)
        print(f"Test Loss: {test_loss:.4f}")