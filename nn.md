# PyTorch 神经网络基础：`__init__` 与 `forward` 方法详解

在 PyTorch 中，自定义神经网络类通常继承自 `nn.Module`，核心方法是 `__init__` 和 `forward`。下面以一个简单的全连接网络为例进行讲解。

---

## 1. 模型代码示例

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义网络结构
        self.fc1 = nn.Linear(28*28, 128)  # 全连接层1
        self.relu = nn.ReLU()             # 激活函数
        self.fc2 = nn.Linear(128, 10)    # 全连接层2
        self.softmax = nn.Softmax(dim=1) # 输出概率归一化

    def forward(self, x):
        # 前向计算流程
        x = x.view(-1, 28*28)  # 展平输入为 (batch_size, 784)
        x = self.fc1(x)        # 第一个全连接层
        x = self.relu(x)       # ReLU 激活
        x = self.fc2(x)        # 第二个全连接层
        x = self.softmax(x)    # Softmax 转为概率分布
        return x

# 创建模型实例
model = SimpleNN()
```
## 2. 方法详解

### 2.1 `__init__` 方法

**作用**：定义网络的各层，包括线性层、卷积层、激活函数等。

**特点**：

- 仅声明网络结构，不进行前向计算。
- 注册子模块，便于 PyTorch 自动管理参数。

---

### 2.2 `forward` 方法

**作用**：定义数据的前向传播流程。

**特点**：

- 输入 $x$ 会依次经过各个层，输出最终结果。
- PyTorch 自动重载 `__call__` 方法，调用模型实例时会触发 `forward`。

**示例**：

```python
output = model(input_tensor)
```
- 相当于执行：

```python
model.forward(input_tensor)
```
- 无需手动调用 forward 方法。
## 3. 数据流说明

1. 输入图像张量 $x$ 展平成 $(\text{batch\_size}, 784)$。
2. 经过第一个全连接层 `fc1`，输出 $(\text{batch\_size}, 128)$。
3. 通过 ReLU 激活函数，增加非线性。
4. 经过第二个全连接层 `fc2`，输出 $(\text{batch\_size}, 10)$。
5. 通过 Softmax 将输出转为概率分布，适合分类任务。

这种结构是典型的全连接神经网络（MLP）分类模型。
## 4. 补充说明

- `nn.Linear(in_features, out_features)`：创建全连接层，将输入特征维度 $in\_features$ 映射到输出维度 $out\_features$。
- `nn.ReLU()`：激活函数，增加网络非线性能力。
- `nn.Softmax(dim=1)`：对指定维度做归一化，使输出值可以看作概率分布。
- `x.view(-1, 28*28)`：将输入张量展平成二维，`-1` 表示自动计算 batch size。
- 在 PyTorch 中，所有 `nn.Module` 的子模块都会自动注册为模型参数，无需手动管理。
