# PyTorch 激活函数详解

本教程介绍常用激活函数，公式可以在 GitHub 上渲染。

---

## 1. ReLU

数学公式：

$$
f(x) = \max(0, x)
$$

解释：

- 将小于0的输入置0，大于0保持不变  
- 计算简单，常用于卷积层或全连接层后

```python
import torch
import torch.nn as nn

relu = nn.ReLU()
x = torch.tensor([[-1.0, 0.0, 2.0]])
y = relu(x)
print("ReLU 输出:\n", y)
