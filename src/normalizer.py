import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def apply_scaler(scaler_name: str, x: torch.Tensor, scaler=None):
    scaler_name = scaler_name.lower()
    x_np = x.detach().cpu().numpy().reshape(-1, 1)

    # train==False
    if scaler is not None:
        if scaler_name == "log":
            x_scaled = np.log(x_np)
        elif scaler_name == "log_zscore":
            x_scaled = scaler.transform(np.log(x_np))
        elif scaler_name == "zscore" or scaler_name == "minmax":
            x_scaled = scaler.transform(x_np)
        else:
            raise ValueError(f"Unsupported scaler name: {scaler_name}")

        return torch.from_numpy(x_scaled.astype(np.float32))

    # train==True
    if scaler_name == "zscore":
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_np)
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x_np)
    elif scaler_name == "log":
        x_scaled = np.log(x_np)
        scaler = None
    elif scaler_name == "log_zscore":
        x_log = np.log(x_np)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_log)
    else:
        raise ValueError(f"Unsupported scaler name: {scaler_name}")

    return torch.from_numpy(x_scaled.astype(np.float32)), scaler