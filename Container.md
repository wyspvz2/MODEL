# PyTorch 容器（Container）详解

PyTorch 提供了一些容器类，用于组织和管理多个子模块，常用的有 `Sequential`、`ModuleList`、`ModuleDict`。

---

**1. nn.Sequential**

- 功能：将多个子模块按顺序组合成一个整体，前一个模块的输出作为下一个模块的输入
- 使用场景：简单的前向顺序网络，如 MLP 或简单 CNN

**示例代码**：

```python
import torch
import torch.nn as nn

model_seq = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

x = torch.randn(2, 10)
y = model_seq(x)
print("Sequential 输出形状:", y.shape)
```
**2. nn.ModuleList**

- **功能**：保存任意数量的子模块的列表，但不会定义前向计算的顺序，需要在 `forward` 中手动调用
- **使用场景**：动态网络结构、多分支网络

```python
import torch
import torch.nn as nn

layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
x = torch.randn(2, 10)
for layer in layers:
    x = layer(x)
print("ModuleList 输出形状:", x.shape)
```
**3. nn.ModuleDict**

- **功能**：以字典形式保存子模块，便于按名字访问
- **使用场景**：多分支或命名网络结构

```python
import torch
import torch.nn as nn

layer_dict = nn.ModuleDict({
    'fc1': nn.Linear(10, 20),
    'relu': nn.ReLU(),
    'fc2': nn.Linear(20, 5)
})

x = torch.randn(2, 10)
x = layer_dict['fc1'](x)
x = layer_dict['relu'](x)
y = layer_dict['fc2'](x)
print("ModuleDict 输出形状:", y.shape)

