import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from src.dataset import PorousDataset
from src.model import CNN
from src.train_val_test import TrainValTest

# --- mlflow ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_experiment")
print(mlflow.get_tracking_uri())

with mlflow.start_run(run_name="CNN"):
    # --- DataLoader ---
    DATA_DIR = "/mnt/c/Users/onion/Documents/slice_data_x_y_z_1"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    TRAIN_M_PATH = os.path.join(DATA_DIR, "target/m_train.pt")
    VAL_M_PATH = os.path.join(DATA_DIR, "target/m_val.pt")
    TEST_M_PATH = os.path.join(DATA_DIR, "target/m_test.pt")
    TRAIN_KAPPA_PATH = os.path.join(DATA_DIR, "target/kappa_train.pt")
    VAL_KAPPA_PATH = os.path.join(DATA_DIR, "target/kappa_val.pt")
    TEST_KAPPA_PATH = os.path.join(DATA_DIR, "target/kappa_test.pt")

    num_train = 1000 # max 63000
    num_val = 1000 # max 13500
    num_test = 1000 # max 13500
    batch_size = 32
    num_workers = 4

    train_dataset = PorousDataset(TRAIN_DIR, TRAIN_M_PATH, TRAIN_KAPPA_PATH, nums_data=num_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = PorousDataset(VAL_DIR, VAL_M_PATH, VAL_KAPPA_PATH, nums_data=num_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = PorousDataset(TEST_DIR, TEST_M_PATH, TEST_KAPPA_PATH, nums_data=num_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Training and Validation and Test---
    learning_rate = 0.001
    num_epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    runner = TrainValTest(train_loader, val_loader, test_loader, model, criterion, optimizer, device, num_epochs=num_epochs)
    runner.train_val()
    runner.test()

    # --- mlflow logging ---
    mlflow.log_param("num_train", num_train)
    mlflow.log_param("num_val", num_val)
    mlflow.log_param("num_test", num_test)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_workers", num_workers)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("device", device)
    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.log_param("criterion", criterion.__class__.__name__)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)

    mlflow.pytorch.log_model(model, "model") # type: ignore
