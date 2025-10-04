# PyTorch 数据增强示例：augment_transform 逐步讲解

本文详细解释下面这个 `augment_transform` 的每一步（含灰度图和 RGB 示例），并给出具体的张量（数值）例子，方便理解变换在管道中如何作用。

```python
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 随机水平翻转（默认概率 p=0.5）
    transforms.RandomRotation(15),         # 随机旋转，角度在 [-15°, +15°] 区间
    transforms.ToTensor(),                 # 转为张量并把像素值从 [0,255] 映射到 [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))   # 归一化到 [-1,1]
])
```

---

## 变换概览（简短）
- **RandomHorizontalFlip(p=0.5)**：以概率 `p` 随机做水平翻转（左右镜像）。默认 `p=0.5`。  
  - 对一个 2×2 灰度矩阵 `[[a,b],[c,d]]`，水平翻转后变为 `[[b,a],[d,c]]`（列顺序颠倒）。
- **RandomRotation(15)**：随机采样角度 `θ ∼ Uniform(-15, +15)`（单位：度），对图像做以中心为旋转点的旋转。对于小角度（如 ±15°），会发生像素插值（双线性或最近邻），输出像素为浮点插值值。注意：`RandomRotation` 的内部实现会处理边界像素填充/裁剪（默认 `expand=False`），所以输出通常仍是原尺寸，但像素被插值和混合。
- **ToTensor()**：将 `PIL.Image` 或 `numpy.uint8` 图像转为 `torch.FloatTensor`，并把像素值从整数 `[0,255]` 缩放到浮点 `[0.0,1.0]`，输出通道顺序为 `(C, H, W)`。
- **Normalize(mean, std)**：对每个通道执行 `out = (x - mean) / std`。常见用法 `mean=0.5,std=0.5` 会把 `[0,1]` 线性映射到 `[-1,1]`（公式简化为 `2x - 1`）。

---

## 1) 灰度图（单通道）逐步示例（数值）
原始 2×2 灰度图（uint8）:
```
[[   0, 128],
 [  64, 255]]
```
记作矩阵元素 `a=0, b=128, c=64, d=255`，也写成 `[[a,b],[c,d]]`。

### A. 情况 1：**不翻转、不旋转（或旋转角度为 0°）**（为了演示 ToTensor/Normalize 的数值）
- **原始 → ToTensor()**（除以 255，变为浮点并调整为 `(1, H, W)`）：
```
tensor([[[0.0000, 0.5020],
         [0.2510, 1.0000]]])
```
数值计算示例：`128/255 ≈ 0.50196078 ≈ 0.5020`，`64/255 ≈ 0.25098039 ≈ 0.2510`。

- **Normalize((0.5,), (0.5,))**（公式 `out = (x - 0.5)/0.5 = 2x - 1`）：
```
tensor([[[-1.0000,  0.0039],
         [-0.4980,  1.0000]]])
```
对应逐像素：
- `0.0000 → -1.0`
- `0.50196 → ≈ 0.00392`
- `0.25098 → ≈ -0.49804`
- `1.0000 → 1.0`

### B. 情况 2：**水平翻转**（示例：RandomHorizontalFlip 触发）
- 翻转后像素矩阵变为：
```
[[128,   0],
 [255,  64]]
```
- **ToTensor()** 结果：
```
tensor([[[0.5020, 0.0000],
         [1.0000, 0.2510]]])
```
- **Normalize**（`2x - 1`）结果：
```
tensor([[[ 0.0039, -1.0000],
         [ 1.0000, -0.4980]]])
```
（对应于每个位置把上面的 `ToTensor` 值代入 `2x-1`）

### C. 情况 3：**旋转 90°（仅作“可视化”说明）**  
> 注意：`RandomRotation(15)` 不会产生 90°，这里只用 90° 举例说明旋转如何改变像素排列（整角度示例，免去插值细节）。

- 90° 逆时针旋转（`np.rot90` 的效果）对原矩阵 `[[a,b],[c,d]]` 的结果是 `[[b,d],[a,c]]`，因此：
```
[[128, 255],
 [  0,  64]]
```
- **ToTensor()**：
```
tensor([[[0.5020, 1.0000],
         [0.0000, 0.2510]]])
```
- **Normalize**：
```
tensor([[[ 0.0039,  1.0000],
         [-1.0000, -0.4980]]])
```

> 小结（灰度图）：RandomHorizontalFlip 只改变列顺序（左右镜像）；RandomRotation 会把像素坐标做连续旋转，若角度不是整倍 90°，通常会产生插值后的浮点像素值（非整数），随后 ToTensor 将其缩到 `[0,1]` 再 Normalize 到 `[-1,1]`。

---

## 2) RGB 图（三通道）逐步示例

为了演示，对下列 2×2 的 RGB 输入（uint8）进行说明：

```
[[ [  0,   0,   0],   [255, 128,  64] ],
 [ [128, 255,   0],   [ 64,  64, 255] ]]
```
把它写为四个像素：`p00=[0,0,0], p01=[255,128,64], p10=[128,255,0], p11=[64,64,255]`。

### A. 不翻转、不旋转（示例）
- **ToTensor()**（通道维度 `C=3`，每通道除以 255）：
```
tensor([[[0.0000, 1.0000],
         [0.5020, 0.2510]],    # R 通道

        [[0.0000, 0.5020],
         [1.0000, 0.2510]],    # G 通道

        [[0.0000, 0.2510],
         [0.0000, 1.0000]]])   # B 通道
```
对应每个通道：R 分别是 `0/255, 255/255, 128/255, 64/255`，以此类推。

- **Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))**（对每个通道执行 `2x-1`）：
```
tensor([[[ -1.0000,  1.0000],
         [  0.0039, -0.4980]],    # R

        [[ -1.0000,  0.0039],
         [  1.0000, -0.4980]],    # G

        [[ -1.0000, -0.4980],
         [ -1.0000,  1.0000]]])   # B
```

### B. 水平翻转（示例：触发 RandomHorizontalFlip）
- 原像素左右镜像后位置变换：`p00 <-> p01`, `p10 <-> p11` （每行列顺序颠倒）。
- 翻转后 **ToTensor**（数值只是每行颠倒后的 values 除以 255），**Normalize** 仍按 `2x-1` 逐通道执行。

### C. 随机旋转（示例说明）
- 对 RGB 图像，旋转会对三个通道同时做相同的几何变换，插值会在每个通道上独立进行，结果仍为浮点值（接着被 `ToTensor`/`Normalize` 处理）。

---

## 3) 可用于本地验证的 Python 示例（不依赖外部输出）

下面代码演示如何**多次**对同一张图片应用 `augment_transform`（或把 `ToTensor` 与 `Normalize` 分开以打印中间结果），以观察随机增强的不同输出：

```python
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms

# 设置可复现的随机种子（对 Python 的 random 生效）
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# 原始 2x2 灰度图（uint8）
arr = np.array([[0, 128],
                [64, 255]], dtype=np.uint8)
img = Image.fromarray(arr, mode='L')

# 定义分步变换：增强（不含 Normalize） + Normalize 单独处理
augment_no_norm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
normalize = transforms.Normalize((0.5,), (0.5,))

# 多次应用看差异
for i in range(4):
    t = augment_no_norm(img)  # torch tensor in [0,1], shape (1, H, W)
    t_norm = normalize(t)
    print(f"Run {i+1}: ToTensor() =>\\n{t.numpy()}\\nNormalize() =>\\n{t_norm.numpy()}\\n")
```

> 运行后你会看到不同 `Run i` 的 `ToTensor()` 输出不同（如果随机操作触发），随后 `Normalize()` 把它们映射到 `[-1,1]`。若你希望完全可复现（即每次都得到相同结果），可以在每次循环前调用 `random.seed(some_fixed_value)`。

---

## 4) 小结（工程实践建议）
- **训练时**：把 `augment_transform` 放在训练集，增加数据多样性，提高模型泛化能力。  
- **验证/测试时**：不要使用随机增强；使用确定性的 `ToTensor()` + `Normalize()` 保证结果稳定。  
- 若需要可复现的增强结果（便于调试/对比实验），控制 Python 的 `random.seed()`（和 `numpy`/`torch` 的 seed），并在每次调用前设置相同种子。

---

## 参考（快速提示）
- `RandomHorizontalFlip` 常用 `p=0.5`（默认）。  
- `RandomRotation(degrees)` 支持传入单个数（表示 `[-degrees, +degrees]`）或二元组`(min, max)`。  
- `ToTensor()` 将 `PIL.Image` 或 `np.uint8` 数组转换为浮点 `torch.Tensor`，并把像素值归一化到 `[0,1]`。  
- `Normalize(mean, std)` 的 mean/std 长度应与通道数一致（灰度图使用单元素元组 `(m,)`，RGB 使用 `(mR,mG,mB)`）。
