import os
import glob
import re
import torch
from torch.utils.data import Dataset

class PorousDataset(Dataset):
    def __init__(self, X_dir, y_path, nums_data):
        files = glob.glob(os.path.join(X_dir, "*.pt"))
        self.X_files = sorted(files, key=lambda files_path: int(re.findall(r'\d+', os.path.basename(files_path))[0]))[:nums_data]
        self.y = torch.load(y_path)[:nums_data]
    
    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, idx):
        X = torch.load(self.X_files[idx])
        y = self.y[idx]
        return X, y