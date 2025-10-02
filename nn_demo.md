# PyTorch 手写体图像识别教学

```python
# 本教程演示如何使用 PyTorch 实现手写数字识别任务（MNIST）。
# 完整流程包括数据加载、模型定义、训练和评估。
# 所有文字说明和代码都放在代码块里，方便 GitHub 显示。

# ==============================
# 1. 导入必要库
# ==============================
# 在开始之前，需要导入 PyTorch 核心库、torchvision 数据集工具以及 DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================
# 2. 数据预处理与加载
# ==============================
# MNIST 图片为 28x28 灰度图，需要将图片转换为张量并归一化到 [-1, 1]。
# 使用 DataLoader 可以按批次加载数据，并支持 shuffle 功能，以打乱训练顺序。
# 这样可以保证每个训练 epoch 都能看到不同顺序的数据，有助于模型收敛。
transform = transforms.Compose([
    transforms.ToTensor(),                # 转换为张量
    transforms.Normalize((0.5,), (0.5,)) # 归一化到 [-1,1]
])

# 下载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 使用 DataLoader 按批次加载数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==============================
# 3. 定义前馈神经网络
# ==============================
# 网络结构包括输入层、隐藏层和输出层：
# 输入层：28x28=784 个神经元
# 隐藏层：128 个神经元，使用 ReLU 激活
# 输出层：10 个神经元（对应数字 0~9 分类），使用 Softmax 输出概率
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 输入层到隐藏层
        self.relu = nn.ReLU()              # 隐藏层 ReLU 激活
        self.fc2 = nn.Linear(128, 10)     # 隐藏层到输出层
        self.softmax = nn.Softmax(dim=1)  # 输出层 Softmax 概率

    def forward(self, x):
        # 前向传播过程
        # x: batch_size x 1 x 28 x 28
        x = x.view(-1, 28*28)  # 展平图片为向量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = SimpleNN()

# ==============================
# 4. 定义损失函数和优化器
# ==============================
# 损失函数：多分类交叉熵
# 优化器：随机梯度下降 (SGD)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ==============================
# 5. 模型训练流程
# ==============================
# 训练流程包含以下步骤：
# 1. 前向传播：计算模型预测输出
# 2. 计算损失：使用交叉熵损失函数
# 3. 反向传播：计算梯度
# 4. 参数更新：通过优化器调整模型参数
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ==============================
# 6. 测试模型准确率
# ==============================
# 在测试集上评估模型性能
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 说明：
# - 本示例展示了一个完整的 MNIST 手写数字识别流程
# - 数据预处理、网络定义、训练和评估全部包含
# - 所有文字描述都在代码块中，便于 GitHub 显示和复制
