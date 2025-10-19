import torch.nn as nn

def get_criterion(criterion_name: str):
    criterion_name = criterion_name.lower()
    if criterion_name == "mse":
        return nn.MSELoss()
    elif criterion_name == "l1":
        return nn.L1Loss()
    elif criterion_name == "smoothl1":
        return nn.SmoothL1Loss()
    elif criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif criterion_name == "bce":
        return nn.BCELoss()
    elif criterion_name == "bcewithlogits":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")