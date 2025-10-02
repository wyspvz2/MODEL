import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import CNN
from utils.dataset import get_dataloaders

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(batch_size=64)
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}] finished, Avg Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("模型已保存到 mnist_cnn.pth")

if __name__ == "__main__":
    train()
