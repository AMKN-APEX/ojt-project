import torch

def get_metrics(metrics_name: str, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    metrics_name = metrics_name.lower()

    # flatten
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    if metrics_name == "r2":
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2.item()

    elif metrics_name == "mse":
        mse = torch.mean((y_true - y_pred) ** 2)
        return mse.item()

    elif metrics_name == "rmse":
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
        return rmse.item()

    elif metrics_name == "mae":
        mae = torch.mean(torch.abs(y_true - y_pred))
        return mae.item()

    elif metrics_name == "mape":
        mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return mape.item()

    else:
        raise ValueError(f"Unsupported criterion: {metrics_name}")