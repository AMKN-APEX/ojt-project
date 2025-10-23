import torch
from dataclasses import dataclass
from typing import Optional, Any
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler
import mlflow

from .metrics import get_metrics

@dataclass
class TrainValTest:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer
    device: torch.device
    metrics_name: str
    num_epochs: int = 20
    scheduler: Optional[Any] = None


    def __post_init__(self):
        self.model.to(self.device)
        print(self.device)


    def run_epoch(self, loader: DataLoader, train: bool = True) -> tuple:
        loss1_total = 0.0
        loss2_total = 0.0
        metric1_total = 0.0
        metric2_total = 0.0
        num_samples = 0

        self.model.train(train)
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device, dtype=torch.float32)
                y1 = y[0].to(self.device, dtype=torch.float32).view(-1, 1) # m
                y2 = y[1].to(self.device, dtype=torch.float32).view(-1, 1) # kappa

                outputs = self.model(x)
                outputs1 = outputs[:, 0].view(-1, 1)
                outputs2 = outputs[:, 1].view(-1, 1)

                loss1 = self.criterion(outputs1, y1)
                loss2 = self.criterion(outputs2, y2)
                loss = loss1 + loss2

                metrics1 = get_metrics(self.metrics_name, y1, outputs1)
                metrics2 = get_metrics(self.metrics_name, y2, outputs2)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_size = x.size(0)
                loss1_total += loss1.item() * batch_size
                loss2_total += loss2.item() * batch_size
                metric1_total += metrics1 * batch_size
                metric2_total += metrics2 * batch_size
                num_samples += batch_size

        avg_loss1 = loss1_total / num_samples
        avg_loss2 = loss2_total / num_samples
        avg_metric1 = metric1_total / num_samples
        avg_metric2 = metric2_total / num_samples

        return avg_loss1, avg_loss2, avg_metric1, avg_metric2


    def train_val(self):
        for epoch in range(self.num_epochs):
            lr = self.optimizer.param_groups[0]['lr']

            train_loss_m, train_loss_k, train_metrics_m, train_metrics_k = self.run_epoch(self.train_loader, train=True)
            val_loss_m, val_loss_k, val_metrics_m, val_metrics_k = self.run_epoch(self.val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss m: {train_loss_m:.4f}, Train Loss kappa: {train_loss_k:.4f}, Val Loss m: {val_loss_m:.4f}, Val Loss kappa: {val_loss_k:.4f}")
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train metrics m: {train_metrics_m:.3f}, Train metrics kappa: {train_metrics_k:.3f}, Val metrics m: {val_metrics_m:.3f}, Val metrics kappa: {val_metrics_k:.3f}")
            
            # MLflowにログ
            mlflow.log_metric("train_loss_m", train_loss_m, step=epoch)
            mlflow.log_metric("train_loss_k", train_loss_k, step=epoch)
            mlflow.log_metric("val_loss_m", val_loss_m, step=epoch)
            mlflow.log_metric("val_loss_k", val_loss_k, step=epoch)
            mlflow.log_metric("lr", lr, step=epoch)
            mlflow.log_metric("train_metrics_m", train_metrics_m, step=epoch)
            mlflow.log_metric("train_metrics_kappa", train_metrics_k, step=epoch)
            mlflow.log_metric("val_metrics_m", val_metrics_m, step=epoch)
            mlflow.log_metric("val_metrics_kappa", val_metrics_k, step=epoch)


    def test(self):
        test_loss_m, test_loss_k, test_metrics_m, test_metrics_k = self.run_epoch(self.test_loader, train=False)
        print(f"Test Loss m: {test_loss_m:.4f}, Test Loss kappa: {test_loss_k:.4f}")
        print(f"Test metrics m: {test_metrics_m:.4f}, Test metrics kappa: {test_metrics_k:.4f}")

        # MLflowにログ
        mlflow.log_metric("test_loss_m", test_loss_m)
        mlflow.log_metric("test_loss_k", test_loss_k)
        mlflow.log_metric("test_metrics_m", test_metrics_m)
        mlflow.log_metric("test_metrics_kappa", test_metrics_k)