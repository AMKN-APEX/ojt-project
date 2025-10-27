import torch
from dataclasses import dataclass
from typing import Optional, Any
from torch.utils.data import DataLoader
from torch import nn, optim
import mlflow

from .metrics import get_metrics
from .normalizer import inverse_scaler

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
    m_scaler_name: str
    kappa_scaler_name: str
    m_scaler: Optional[Any] = None
    kappa_scaler: Optional[Any] = None
    num_epochs: int = 20
    scheduler: Optional[Any] = None


    def __post_init__(self):
        self.model.to(self.device)
        print(self.device)


    def run_epoch(self, loader: DataLoader, train: bool = True) -> tuple:
        loss_m_total = 0.0
        loss_k_total = 0.0
        num_samples = 0

        self.model.train(train)
        with torch.set_grad_enabled(train):
            y_m_list = []
            y_k_list = []
            out_m_list = []
            out_k_list = []

            for x, y in loader:
                x = x.to(self.device, dtype=torch.float32)
                y_m = y[0].to(self.device, dtype=torch.float32).view(-1, 1) # m
                y_k = y[1].to(self.device, dtype=torch.float32).view(-1, 1) # kappa

                outputs = self.model(x)
                outputs_m = outputs[:, 0].view(-1, 1)
                outputs_k = outputs[:, 1].view(-1, 1)

                loss_m = self.criterion(outputs_m, y_m)
                loss_k = self.criterion(outputs_k, y_k)
                loss = loss_m + loss_k

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_size = x.size(0)
                loss_m_total += loss_m.item() * batch_size
                loss_k_total += loss_k.item() * batch_size

                y_m_list.append(y_m.detach().cpu())
                y_k_list.append(y_k.detach().cpu())
                out_m_list.append(outputs_m.detach().cpu())
                out_k_list.append(outputs_k.detach().cpu())
                num_samples += batch_size

        avg_loss_m = loss_m_total / num_samples
        avg_loss_k = loss_k_total / num_samples

        y_m_all = torch.cat(y_m_list, dim=0)
        out_m_all = torch.cat(out_m_list, dim=0)
        y_k_all = torch.cat(y_k_list, dim=0)
        out_k_all = torch.cat(out_k_list, dim=0)

        y_m_all = inverse_scaler(self.m_scaler_name, y_m_all, self.m_scaler)
        out_m_all = inverse_scaler(self.m_scaler_name, out_m_all, self.m_scaler)
        y_k_all = inverse_scaler(self.kappa_scaler_name, y_k_all, self.kappa_scaler)
        out_k_all = inverse_scaler(self.kappa_scaler_name, out_k_all, self.kappa_scaler)

        avg_metric_m = get_metrics(metrics_name=self.metrics_name, y_true=y_m_all, y_pred=out_m_all)
        avg_metric_k = get_metrics(metrics_name=self.metrics_name, y_true=y_k_all, y_pred=out_k_all)

        return avg_loss_m, avg_loss_k, avg_metric_m, avg_metric_k


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