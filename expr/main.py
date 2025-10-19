import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from omegaconf import OmegaConf 

# 親ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import PorousDataset
from src.model import CNN
from src.train_val_test import TrainValTest

# --- Load config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cfg_path = os.path.join(BASE_DIR, "config.yaml")
cfg = OmegaConf.load(cfg_path)

# --- mlflow ---
mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
mlflow.set_experiment(cfg.mlflow.experiment_name)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name=cfg.mlflow.run_name):
    # --- DataLoader ---
    DATA_DIR = cfg.data.directory.DATA_DIR
    TRAIN_DIR = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[1])
    VAL_DIR = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[2])
    TEST_DIR = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[3])
    TRAIN_M_PATH = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[0], cfg.data.directory.M_NAMES[0])
    VAL_M_PATH = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[0], cfg.data.directory.M_NAMES[1])
    TEST_M_PATH = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[0], cfg.data.directory.M_NAMES[2])
    TRAIN_KAPPA_PATH = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[0], cfg.data.directory.KAPPA_NAMES[0])
    VAL_KAPPA_PATH = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[0], cfg.data.directory.KAPPA_NAMES[1])
    TEST_KAPPA_PATH = os.path.join(DATA_DIR, cfg.data.directory.SUB_DATA_DIRS[0], cfg.data.directory.KAPPA_NAMES[2])

    num_train = cfg.data.data_loader.num_train
    num_val = cfg.data.data_loader.num_val
    num_test = cfg.data.data_loader.num_test
    batch_size = cfg.data.data_loader.batch_size
    num_workers = cfg.data.data_loader.num_workers

    train_dataset = PorousDataset(TRAIN_DIR, TRAIN_M_PATH, TRAIN_KAPPA_PATH, nums_data=num_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = PorousDataset(VAL_DIR, VAL_M_PATH, VAL_KAPPA_PATH, nums_data=num_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = PorousDataset(TEST_DIR, TEST_M_PATH, TEST_KAPPA_PATH, nums_data=num_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Training and Validation and Test---
    learning_rate = cfg.train.learning_rate
    num_epochs = cfg.train.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    runner = TrainValTest(train_loader, val_loader, test_loader, model, criterion, optimizer, device, num_epochs=num_epochs)
    runner.train_val()
    runner.test()

    # --- mlflow logging ---
    mlflow.log_param("data_directory", DATA_DIR)
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

    # mlflow.pytorch.log_model(model, name="model") # type: ignore
