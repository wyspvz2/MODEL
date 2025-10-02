from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# =========================
# 固定变换示例
# =========================
fixed_transform = transforms.Compose([
    transforms.ToTensor(),                  # 转为张量
    transforms.Normalize((0.5,), (0.5,))   # 归一化到 [-1,1]
])

mnist_fixed = datasets.MNIST(root='./data', train=True, download=True, transform=fixed_transform)
loader_fixed = DataLoader(mnist_fixed, batch_size=64, shuffle=True)

# 每次取出的图片内容固定，只是转成张量和归一化
x_fixed, y_fixed = next(iter(loader_fixed))
print("固定变换 batch_x shape:", x_fixed.shape)
print("固定变换 batch_y shape:", y_fixed.shape)

# =========================
# 随机增强示例
# =========================
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomRotation(15),         # 随机旋转 ±15 度
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_aug = datasets.MNIST(root='./data', train=True, download=True, transform=augment_transform)
loader_aug = DataLoader(mnist_aug, batch_size=64, shuffle=True)

# 每次取出的图片可能不同，训练时增加数据多样性
x_aug, y_aug = next(iter(loader_aug))
print("随机增强 batch_x shape:", x_aug.shape)
print("随机增强 batch_y shape:", y_aug.shape)
