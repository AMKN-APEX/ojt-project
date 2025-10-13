import os
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# --- Dataset ---
DATA_DIR = "/mnt/c/Users/onion/Documents/slice_data_x_y_z_1"
X_DIR = os.path.join(DATA_DIR, "train")
Y_PATH = os.path.join(DATA_DIR, "target/m_train.pt")

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

dataset = PorousDataset(X_DIR, Y_PATH, nums_data=1000)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# --- Model ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(32 * 46 * 46, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Training and Validation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.view(-1, 1)  # [batch, 1] の形に揃える
        
        # --- 順伝播 ---
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # --- 逆伝播 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}") 

# --- Test ---


# --- Visualization ---

