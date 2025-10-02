# torchvision.datasets 使用方法

`torchvision.datasets` 提供了常用视觉数据集接口，可以方便地下载和加载数据集。

## 示例代码

```python
from torchvision import datasets

# 下载 MNIST 训练集
mnist_train = datasets.MNIST(root='./data', train=True, download=True)

# 下载 CIFAR-10 训练集
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True)
