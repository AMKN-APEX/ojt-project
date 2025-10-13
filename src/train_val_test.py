import torch

def train_val(model, train_loader, val_loader, train_dataset, val_dataset, device, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_train, y_train in train_loader:
            X_train = X_train.to(device, dtype=torch.float32)
            y_train = y_train.to(device, dtype=torch.float32)
            y_train = y_train.view(-1, 1)

            train_outputs = model(X_train)
            loss = criterion(train_outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_train.size(0)

        train_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device, dtype=torch.float32)
                y_val = y_val.to(device, dtype=torch.float32)
                y_val = y_val.view(-1, 1)

                val_outputs = model(X_val)
                loss = criterion(val_outputs, y_val)

                val_loss += loss.item() * X_val.size(0)

        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def test(model, test_loader, test_dataset, device, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device, dtype=torch.float32)
            y_test = y_test.to(device, dtype=torch.float32)
            y_test = y_test.view(-1, 1)

            test_outputs = model(X_test)
            loss = criterion(test_outputs, y_test)

            test_loss += loss.item() * X_test.size(0)

    test_loss /= len(test_dataset)

    print(f"Test Loss: {test_loss:.4f}")