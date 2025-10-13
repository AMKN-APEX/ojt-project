import os
import glob
import re
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class PorousDataset(Dataset):
    X_dir: str
    y_path: str
    nums_data: int

    def __post_init__(self):
        files = glob.glob(os.path.join(self.X_dir, "*.pt"))
        self.X_files = sorted(files, key=lambda files_path: int(re.findall(r'\d+', os.path.basename(files_path))[0]))[:self.nums_data]

        self.y = torch.load(self.y_path)[:self.nums_data]

    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, idx):
        X = torch.load(self.X_files[idx])
        y = self.y[idx]
        return X, y