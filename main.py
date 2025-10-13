import os
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader

# --- Dataset ---
DATA_DIR = "/mnt/c/Users/onion/Documents/slice_data_x_y_z_1"
X_DIR = os.path.join(DATA_DIR, "train")
Y_PATH = os.path.join(DATA_DIR, "target/m_train.pt")

class PorousDataset(Dataset):
    def __init__(self, X_dir, y_path):
        files = glob.glob(os.path.join(X_dir, "*.pt"))
        self.X_files = sorted(files, key=lambda files_path: int(re.findall(r'\d+', os.path.basename(files_path))[0]))
        self.y = torch.load(y_path)
    
    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, idx):
        X = torch.load(self.X_files[idx])
        y = self.y[idx]
        return X, y

dataset = PorousDataset(X_DIR, Y_PATH)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# --- Model ---


# --- Training and Validation ---


# --- Test ---


# --- Visualization ---

