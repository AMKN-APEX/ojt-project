from dataclasses import dataclass
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def apply_scaler(scaler_name: str, x: torch.Tensor):
    scaler_name = scaler_name.lower()
    x_np = x.detach().cpu().numpy().reshape(-1, 1)

    if scaler_name == "zscore":
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_np)
        return x_scaled
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x_np)
        return x_scaled
    elif scaler_name == "log":
        x_scaled = np.log(x_np)
        return x_scaled
    elif scaler_name == "log_zscore":
        x_log = np.log(x_np)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_log)
        return x_scaled
    else:
        raise ValueError(f"Unsupported optimizer name: {scaler_name}")