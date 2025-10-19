import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
class CNN_256(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
def get_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "cnn":
        return CNN()
    elif model_name == "cnn_256":
        return CNN_256()
    else:
        raise ValueError(f"Unsupported criterion: {model_name}")