# PyTorch DataLoader 教程

在 PyTorch 中，`DataLoader` 是配合 `Dataset` 使用的数据加载工具。它的主要作用是 **批量加载、打乱顺序、并行读取** 数据，方便模型训练。

---

## 1️⃣ DataLoader 的作用

1. **批量加载数据（Batching）**  
   - 训练神经网络通常一次处理多个样本，而不是一条一条处理。  
   - DataLoader 会从 Dataset 中取出 `batch_size` 条样本，组合成一个批次 tensor。  
   - 例如：
     ```
     Dataset 返回：
     dataset[0] -> (x0, y0)
     dataset[1] -> (x1, y1)
     dataset[2] -> (x2, y2)
     dataset[3] -> (x3, y3)

     DataLoader 返回批次（batch_size=4）：
     batch_x -> [[x0], [x1], [x2], [x3]]
     batch_y -> [y0, y1, y2, y3]
     ```

2. **打乱数据顺序（Shuffle）**  
   - 避免模型因为数据集的训练顺序影响训练结果。  
   - 设置 `shuffle=True`，每个 epoch 都会随机打乱数据顺序。

3. **并行读取数据（num_workers）**  
   - 使用多进程或多线程加快数据读取速度。  
   - 对于大规模图片或多模态数据集尤为有用。

4. **自动迭代**  
   - DataLoader 提供迭代器接口，可以直接在训练循环里使用：
     ```text
     for batch_x, batch_y in loader:
         # batch_x, batch_y 已经是一个批次的 tensor
         # 送入模型训练
     ```

---

## 2️⃣ DataLoader 与 Dataset 的关系

| 名称       | 作用                                   | 输出示意 |
|------------|--------------------------------------|-----------|
| Dataset    | 定义每条数据怎么取                      | dataset[0] -> (x0, y0) |
| DataLoader | 批量取数据 + 打乱 + 并行                | batch_x -> [x0, x1,...], batch_y -> [y0, y1,...] |

- **Dataset** = “数据仓库”，负责存储和索引  
- **DataLoader** = “快递员”，负责批量打包和送到模型  

---

## 3️⃣ 多模态数据的 DataLoader

- 如果 Dataset 是多模态 `(图片, 文本, 标签)`：  
  - DataLoader 会把每个模态分别批量组合：
    ```
    batch_img  -> [img0, img1, img2, ...]
    batch_text -> [text0, text1, text2, ...]
    batch_label -> [label0, label1, label2, ...]
    ```
- 模型训练时，每个批次同时包含不同模态的数据，保持对齐。

---

## 4️⃣ 直观理解

- **Dataset** = 仓库，存储单条样本 `(x, y)`  
- **DataLoader** = 快递员，把单条样本打包成批次送到模型训练  
- 每个批次 shape 示例：
  - 特征 tensor: `[batch_size, 特征维度]`
  - 标签 tensor: `[batch_size]`  

---

## 5️⃣ 代码示例

import torch
from torch.utils.data import DataLoader

# 直接导入你写好的 Dataset
from my_dataset import MyDataset

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
Y_data = [0, 1, 0, 1, 0]

# 创建 Dataset 实例
dataset = MyDataset(X_data, Y_data)

# 用 Dataset 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,  # 每批 2 条数据
    shuffle=True,  # 打乱顺序
    num_workers=0  # 单进程加载
)

# 迭代 DataLoader
for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"批次 {batch_idx}:")
    print("batch_x:", batch_x)
    print("batch_y:", batch_y)
    print("-" * 30)

