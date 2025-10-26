import os
import glob
import re
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from .normalizer import apply_scaler

@dataclass
class PorousDataset(Dataset):
    X_dir: str
    m_path: str
    kappa_path: str
    nums_data: int
    m_scaler_name: str
    kappa_scaler_name: str


    def __post_init__(self):
        files = glob.glob(os.path.join(self.X_dir, "*.pt"))
        self.X_files = sorted(files, key=lambda files_path: int(re.findall(r'\d+', os.path.basename(files_path))[0]))[:self.nums_data]

        self.m = torch.load(self.m_path)[:self.nums_data]
        self.kappa = torch.load(self.kappa_path)[:self.nums_data]

        self.m_scaled =apply_scaler(self.m_scaler_name, self.m)
        self.kappa_scaled = apply_scaler(self.kappa_scaler_name, self.kappa)


    def __len__(self):
        return len(self.X_files)


    def __getitem__(self, idx):
        x = torch.load(self.X_files[idx])
        m_scaled = self.m_scaled[idx]
        kappa_scaled = self.kappa_scaled[idx]
        return x, [m_scaled, kappa_scaled]