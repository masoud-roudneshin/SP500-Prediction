import torch
import torch.optim as optim
import torch.nn as nn
# from model import StockPriceLSTM
# from data_loader import create_dataloader

def train_model(train_loader, dataset, epochs=100, lr=0.001, batch_size=64, sequence_length=5):

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model
    model = StockPriceLSTM()
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_train, y_train in train_loader:

            #print(X_train.unsqueeze(2).shape)
            X_train = X_train.unsqueeze(2).to(device).to(torch.float32)
            y_train = y_train.to(device).to(torch.float32)

            # Forward pass
            output = model(X_train)

            loss = criterion(output, y_train.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')


    return model, loss_history
