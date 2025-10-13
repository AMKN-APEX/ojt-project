import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.dataset import PorousDataset
from src.model import CNN
from src.train_val_test import TrainValTest

# --- DataLoader ---
DATA_DIR = "/mnt/c/Users/onion/Documents/slice_data_x_y_z_1"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_Y_PATH = os.path.join(DATA_DIR, "target/m_train.pt")
VAL_Y_PATH = os.path.join(DATA_DIR, "target/m_val.pt")
TEST_Y_PATH = os.path.join(DATA_DIR, "target/m_test.pt")

train_dataset = PorousDataset(TRAIN_DIR, TRAIN_Y_PATH, nums_data=1000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = PorousDataset(VAL_DIR, VAL_Y_PATH, nums_data=1000)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

test_dataset = PorousDataset(TEST_DIR, TEST_Y_PATH, nums_data=1000)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# --- Training and Validation and Test---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

runner = TrainValTest(train_loader, val_loader, test_loader, model, criterion, optimizer, device, num_epochs=3)
runner.train_val()
runner.test()

