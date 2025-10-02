# PyTorch 自定义 Dataset 教程

在 PyTorch 里，如果我们想从自己的数据源创建数据集，需要继承 `torch.utils.data.Dataset`。

Dataset 负责定义 **每条数据怎么取**，DataLoader 负责 **批量加载和打乱** 数据。

---

## 1. Dataset 核心方法

- `__len__(self)`：返回数据集总样本数
- `__getitem__(self, idx)`：根据索引 `idx` 返回一条样本 `(x, y)`

---

## 2. 示例：列表数据

```python
import torch
from torch.utils.data import Dataset

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        初始化数据集
        X_data: 输入特征 (列表或数组)
        Y_data: 标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """返回数据集大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """根据索引返回一条样本"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
Y_data = [1, 0, 1, 0]

# 创建数据集实例
dataset = MyDataset(X_data, Y_data)

# 查看第 0 条数据
print(dataset[0])  # 输出: (tensor([1., 2.]), tensor(1.))
print("数据集大小:", len(dataset))
