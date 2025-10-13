from torch.utils.data import Dataset, DataLoader

# --- Dataset ---
class PorousDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X = X_data
        self.y = y_data
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
dataset = PorousDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)  

# --- Model ---


# --- Training and Validation ---


# --- Test ---


# --- Visualization ---

