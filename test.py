import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from models.resnet import resnet18   # 或 CNN
from utils.dataset import get_dataloaders

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloaders(batch_size=64)

    # 加载模型
    model = resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算准确率
    correct = np.sum(np.array(y_true) == np.array(y_pred))
    acc = 100 * correct / len(y_true)
    print(f"Accuracy on test set: {acc:.2f}%")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Acc={acc:.2f}%)")
    plt.show()


if __name__ == "__main__":
    test()
