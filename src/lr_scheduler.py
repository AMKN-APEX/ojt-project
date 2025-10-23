from typing import Any, Optional, List
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(scheduler_name: str, optimizer: Optimizer, **kwargs) -> Any:
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "steplr":
        step_size = kwargs.pop("step_size", None)
        if step_size is None:
            raise ValueError("StepLR requires 'step_size' argument")
        gamma = kwargs.pop("gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs)

    if scheduler_name == "multistep":
        milestones = kwargs.pop("milestones", None)
        if milestones is None:
            raise ValueError("MultiStepLR requires 'milestones' argument (list of ints)")
        gamma = kwargs.pop("gamma", 0.1)
        if isinstance(milestones, str):
            milestones = [int(x) for x in milestones.split(",")]
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, **kwargs)

    if scheduler_name == "exponential":
        gamma = kwargs.pop("gamma", None)
        if gamma is None:
            raise ValueError("ExponentialLR requires 'gamma' argument")
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **kwargs)

    if scheduler_name == "cosine":
        T_max = kwargs.pop("T_max", None)
        if T_max is None:
            raise ValueError("CosineAnnealingLR requires 'T_max' argument")
        eta_min = kwargs.pop("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, **kwargs)

    if scheduler_name == "cosine_warm_restarts":
        T_0 = kwargs.pop("T_0", None)
        if T_0 is None:
            raise ValueError("CosineAnnealingWarmRestarts requires 'T_0' argument")
        T_mult = kwargs.pop("T_mult", 1)
        eta_min = kwargs.pop("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, **kwargs)

    if scheduler_name == "reduceonplateau":
        mode = kwargs.pop("mode", "min")
        factor = kwargs.pop("factor", 0.1)
        patience = kwargs.pop("patience", 10)
        threshold = kwargs.pop("threshold", 1e-4)
        cooldown = kwargs.pop("cooldown", 0)
        min_lr = kwargs.pop("min_lr", 0)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, cooldown=cooldown, min_lr=min_lr, **kwargs)

    raise ValueError(f"Unsupported scheduler name: {scheduler_name}")
