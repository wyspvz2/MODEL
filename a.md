# PyTorch 激活函数详解

本教程介绍常用激活函数。

---

## 1. ReLU
<img width="473" height="382" alt="image" src="https://github.com/user-attachments/assets/79bf9f4a-fcce-4a9c-bc84-d69f787c2eee" />

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
```

## 2. Sigmoid

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/606014d0-e40a-4c21-a18f-8824ac3a86ea" />

数学公式：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

解释：

- 将输入映射到 (0,1)  
- 常用于二分类输出层

```python
import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()
x = torch.tensor([[-1.0, 0.0, 2.0]])
y = sigmoid(x)
print("Sigmoid 输出:\n", y)
```
## 3. Tanh
<img width="425" height="234" alt="image" src="https://github.com/user-attachments/assets/0a34c0d6-9627-4a65-a04e-82690727ee00" />

数学公式：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

解释：

- 将输入映射到 (-1, 1)  
- 均值为 0，有助于训练收敛  
- 常用于序列模型或隐藏层激活函数

```python
import torch
import torch.nn as nn

tanh = nn.Tanh()
x = torch.tensor([[-1.0, 0.0, 2.0]])
y = tanh(x)
print("Tanh 输出:\n", y)
```
## 4. LeakyReLU
<img width="683" height="559" alt="image" src="https://github.com/user-attachments/assets/1c91f705-c38d-4025-9cbd-d6ebf25da8a0" />

数学公式：

$$
f(x) = 
\begin{cases} 
x, & x > 0 \\
\alpha x, & x \le 0
\end{cases}, \quad \alpha = 0.01
$$

解释：

- 避免 ReLU 的“死亡神经元”问题  
- 对负值仍保留小梯度，不完全置零  
- 常用于卷积层或全连接层激活函数

```python
import torch
import torch.nn as nn

leaky_relu = nn.LeakyReLU(negative_slope=0.01)
x = torch.tensor([[-1.0, 0.0, 2.0]])
y = leaky_relu(x)
print("LeakyReLU 输出:\n", y)
