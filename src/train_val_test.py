import torch

class TrainValTest():
    def __init__(self, train_loader, val_loader, test_loader, model, criterion, optimizer, device, num_epochs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

    def run_epoch(self, loader, train=True):
        loss_total = 0.0
        self.model.train() if train else self.model.eval()
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32).view(-1,1)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                loss_total += loss.item() * x.size(0)
        return loss_total / len(loader.dataset)

    def train_val(self):
        for epoch in range(self.num_epochs):
            train_loss = self.run_epoch(self.train_loader, train=True)
            val_loss = self.run_epoch(self.val_loader, train=False)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def test(self):
        test_loss = self.run_epoch(self.test_loader, train=False)
        print(f"Test Loss: {test_loss:.4f}")