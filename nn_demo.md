# PyTorch 手写体图像识别教学

---

# 1. 导入必要库

```python

# 本教程演示如何使用 PyTorch 实现 MNIST 手写数字识别任务
# 包含数据加载、模型定义、训练和测试
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

```

# 2. 数据预处理与加载
# MNIST 图片为 28x28 灰度图
# 需要将图片转换为张量并归一化到 [-1, 1]
# DataLoader 按批次加载数据，并支持 shuffle 功能
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

# 3. 定义前馈神经网络
# 网络结构：
# 输入层: 28*28 = 784 个神经元
# 隐藏层: 128 个神经元, ReLU 激活
# 输出层: 10 个神经元, Softmax 输出概率
```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播
        x = x.view(-1, 28*28)  # 展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = SimpleNN()
```

# 4. 定义损失函数和优化器
# 多分类交叉熵损失
# 优化器: 随机梯度下降 (SGD)
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

# 5. 模型训练
# 训练流程:
# 1. 前向传播: 计算预测输出
# 2. 计算损失
# 3. 反向传播
# 4. 参数更新
```python
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```

# 6. 测试模型准确率
# 在测试集上评估模型性能
```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```
