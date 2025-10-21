from dataclasses import dataclass
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler

@dataclass
class Normalizer:
    scaler_name: str
    x: torch.Tensor


    def apply_scaler(self):
        scaler_name = self.scaler_name.lower()
        x = self.x.detach().cpu().numpy().reshape(-1, 1)

        if scaler_name == "zscore":
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            return x_scaled
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x)
            return x_scaled
        elif scaler_name == "log":
            x_scaled = np.log(x)
            return x_scaled
        elif scaler_name == "log_zscore":
            x_log = np.log(x)
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_log)
            return x_scaled
        else:
            raise ValueError(f"Unsupported optimizer name: {scaler_name}")