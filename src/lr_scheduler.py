from typing import Any, Optional, List
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(scheduler_name: str, optimizer: Optimizer, params: dict) -> Any:
    if scheduler_name is None:
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "steplr":
        step_size = params["step_size"]
        gamma = params["gamma"]
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if scheduler_name == "multistep":
        milestones = params["milestones"]
        gamma = params["gamma"]
        if isinstance(milestones, str):
            milestones = [int(x) for x in milestones.split(",")]
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if scheduler_name == "exponential":
        gamma = params["gamma"]
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if scheduler_name == "cosine":
        T_max = params["T_max"]
        eta_min = params["eta_min"]
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    if scheduler_name == "cosine_warm_restarts":
        T_0 = params["T_0"]
        T_mult = params["T_mult"]
        eta_min = params["eta_min"]
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    if scheduler_name == "reduceonplateau":
        mode = params["mode"]
        factor = params["factor"]
        patience = params["patience"]
        threshold = params["threshold"]
        cooldown = params["cooldown"]
        min_lr = params["min_lr"]
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, cooldown=cooldown, min_lr=min_lr)

    raise ValueError(f"Unsupported scheduler name: {scheduler_name}")
