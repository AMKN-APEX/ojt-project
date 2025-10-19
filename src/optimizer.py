import torch.optim as optim


def get_optimizer(optimizer_name: str, model_params, lr: float, **kwargs):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return optim.Adam(model_params, lr=lr, **kwargs)
    elif optimizer_name == "sgd":
        return optim.SGD(model_params, lr=lr, **kwargs)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model_params, lr=lr, **kwargs)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model_params, lr=lr, **kwargs)
    elif optimizer_name == "adamw":
        return optim.AdamW(model_params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")