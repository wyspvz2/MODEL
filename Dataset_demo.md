# PyTorch 自定义 Dataset 教程

在 PyTorch 里，如果我们想从自己的数据源创建数据集，需要继承 `torch.utils.data.Dataset`。

Dataset 负责定义 **每条数据怎么取**，DataLoader 负责 **批量加载和打乱** 数据。

---

# . Dataset 核心方法

- `__len__(self)`：返回数据集总样本数
- `__getitem__(self, idx)`：根据索引 `idx` 返回一条样本 `(x, y)`

---

# PyTorch 自定义 Dataset 示例（全部注释版）

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

# =========================
# 示例 1：列表 / 数组数据
# =========================

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
        """
        返回指定索引的数据
        输出格式: (特征 tensor, 标签 tensor)
        """
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 float32 tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)  # 转换为 float32 tensor
        return x, y

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 特征列表
Y_data = [1, 0, 1, 0]  # 标签列表

# 创建数据集实例
dataset = MyDataset(X_data, Y_data)

# 查看第 0 条数据
print(dataset[0])  # 输出: (tensor([1., 2.]), tensor(1.))
print("数据集大小:", len(dataset))

# =========================
# 示例 2：CSV 文件数据
# =========================

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        """
        初始化 CSV 数据集
        csv_path: CSV 文件路径
        CSV 每行格式: 特征1, 特征2, ..., 标签
        """
        self.data = pd.read_csv(csv_path).values  # 转为 numpy 数组

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回指定索引的数据
        输出格式: (特征 tensor, 标签 tensor)
        """
        row = self.data[idx]
        x = torch.tensor(row[:-1], dtype=torch.float32)  # 特征
        y = torch.tensor(row[-1], dtype=torch.float32)  # 标签
        return x, y

# =========================
# 示例 3：图片数据
# =========================

class ImageDataset(Dataset):
    def __init__(self, image_files, labels):
        """
        初始化图片数据集
        image_files: 图片文件路径列表
        labels: 标签列表
        """
        self.image_files = image_files
        self.labels = labels

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        返回指定索引的数据
        输出格式: (图片 tensor, 标签 tensor)
        """
        img = Image.open(self.image_files[idx]).convert("RGB")  # 打开图片并转为 RGB
        img = torch.tensor(img, dtype=torch.float32)            # 转为 tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# =========================
# 示例 4：多模态数据（图片 + 文本）
# =========================

class MultiModalDataset(Dataset):
    def __init__(self, image_files, text_data, labels):
        """
        初始化多模态数据集
        image_files: 图片文件路径列表
        text_data: 文本数据列表 (可为 token 或 embedding)
        labels: 标签列表
        """
        self.image_files = image_files
        self.text_data = text_data
        self.labels = labels

    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回指定索引的数据
        输出格式: (图片 tensor, 文本 tensor, 标签 tensor)
        """
        img = Image.open(self.image_files[idx]).convert("RGB")  # 图片模态
        img = torch.tensor(img, dtype=torch.float32)            # 转为 tensor
        text = torch.tensor(self.text_data[idx], dtype=torch.float32)  # 文本模态
        label = torch.tensor(self.labels[idx], dtype=torch.float32)    # 标签
        return img, text, label





