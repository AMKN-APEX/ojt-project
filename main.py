import os
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# --- Dataset ---
DATA_DIR = "/mnt/c/Users/onion/Documents/slice_data_x_y_z_1"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
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

train_dataset = PorousDataset(TRAIN_DIR, Y_PATH, nums_data=1000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

val_dataset = PorousDataset(VAL_DIR, Y_PATH, nums_data=1000)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)

test_dataset = PorousDataset(TEST_DIR, Y_PATH, nums_data=1000)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

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
    # --- Training ---
    model.train()
    train_loss = 0.0
    for X_train, y_train in train_loader:
        X_train = X_train.to(device, dtype=torch.float32)
        y_train = y_train.to(device, dtype=torch.float32)
        y_train = y_train.view(-1, 1)

        train_outputs = model(X_train)
        loss = criterion(train_outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_train.size(0)

    train_loss /= len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device, dtype=torch.float32)
            y_val = y_val.view(-1, 1)

            val_outputs = model(X_val)
            loss = criterion(val_outputs, y_val)

            val_loss += loss.item() * X_val.size(0)

    val_loss /= len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# --- Test ---
model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.to(device, dtype=torch.float32)
        y_test = y_test.to(device, dtype=torch.float32)
        y_test = y_test.view(-1, 1)

        test_outputs = model(X_test)
        loss = criterion(test_outputs, y_test)

        test_loss += loss.item() * X_test.size(0)

test_loss /= len(test_dataset)

print(f"Test Loss: {test_loss:.4f}")

# --- Visualization ---

