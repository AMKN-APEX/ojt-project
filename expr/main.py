import os
import sys
import torch
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import hydra
from omegaconf import DictConfig
from timm.models import create_model

# 親ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import PorousDataset
from src.train_val_test import TrainValTest
from src.optimizer import get_optimizer
from src.criterion import get_criterion
from src.lr_scheduler import get_scheduler
from src.utils import log_config_to_mlflow


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    # --- parameters ---
    # data.directory
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

    # data.data_loader
    num_train = cfg.data.data_loader.num_train
    num_val = cfg.data.data_loader.num_val
    num_test = cfg.data.data_loader.num_test
    batch_size = cfg.data.data_loader.batch_size
    num_workers = cfg.data.data_loader.num_workers
    pin_memory = cfg.data.data_loader.pin_memory

    # data
    m_scaler_name = cfg.data.normalizer.m_scaler_name
    kappa_scaler_name = cfg.data.normalizer.kappa_scaler_name
    
    # model
    model_name = cfg.model.model_name
    pretrained = cfg.model.pretrained

    # train
    criterion_name = cfg.train.criterion_name
    optimizer_name = cfg.train.optimizer_name
    learning_rate = cfg.train.learning_rate
    num_epochs = cfg.train.num_epochs

    # train.scheduler
    scheduler_name = cfg.train.scheduler.name
    scheduler_params = cfg.train.scheduler.params

    # mlflow
    tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiment_name
    run_name = cfg.mlflow.run_name

    # --- mlflow ---
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=run_name):
        # --- DataLoader ---
        train_dataset = PorousDataset(TRAIN_DIR, TRAIN_M_PATH, TRAIN_KAPPA_PATH, num_train, m_scaler_name, kappa_scaler_name)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        val_dataset = PorousDataset(VAL_DIR, VAL_M_PATH, VAL_KAPPA_PATH, num_val, m_scaler_name, kappa_scaler_name)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        test_dataset = PorousDataset(TEST_DIR, TEST_M_PATH, TEST_KAPPA_PATH, num_test, m_scaler_name, kappa_scaler_name)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # --- Training and Validation and Test---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(model_name=model_name, pretrained=pretrained)
        criterion = get_criterion(criterion_name)
        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
        scheduler = get_scheduler(scheduler_name, optimizer, scheduler_params)

        runner = TrainValTest(train_loader, val_loader, test_loader, model, criterion, optimizer, device, num_epochs, scheduler)
        runner.train_val()
        runner.test()

        # --- mlflow logging ---
        log_config_to_mlflow(cfg)
        # mlflow.pytorch.log_model(model, name="model") # type: ignore

if __name__ == "__main__":
    main()