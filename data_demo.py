import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# ======================================================
# 1. 自定义 Dataset
# ======================================================
class MyDataset(Dataset):
    """一个简单的自定义数据集：y = x^2"""
    def __init__(self, data_size=10):
        self.data = torch.arange(1, data_size+1, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x ** 2
        return x, y

def demo_custom_dataset():
    print("\n=== 自定义 Dataset 示例 ===")
    dataset = MyDataset(data_size=5)
    for i in range(len(dataset)):
        print(dataset[i])

# ======================================================
# 2. TensorDataset
# ======================================================
def demo_tensor_dataset():
    print("\n=== TensorDataset 示例 ===")
    x = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)  # [[1],[2],[3],[4],[5]]
    y = x ** 2
    dataset = TensorDataset(x, y)
    for data in dataset:
        print(data)

# ======================================================
# 3. DataLoader
# ======================================================
def demo_dataloader():
    print("\n=== DataLoader 示例 ===")
    x = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)
    y = x ** 2
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in loader:
        print(batch)

# ======================================================
# 4. ImageFolder
# ======================================================
def demo_imagefolder():
    print("\n=== ImageFolder 示例 ===")
    # 数据目录结构:
    # dataset/
    #     class1/
    #         img001.png
    #         img002.png
    #     class2/
    #         img101.png
    #         img102.png
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root="dataset", transform=transform)
    print("类别:", dataset.classes)
    print("样本数量:", len(dataset))
    img, label = dataset[0]
    print("图像尺寸:", img.shape, "标签:", label)

# ======================================================
# 主函数
# ======================================================
if __name__ == "__main__":
    demo_custom_dataset()
    demo_tensor_dataset()
    demo_dataloader()
    # 注意: 需要准备 dataset/ 文件夹才能运行 ImageFolder
    # demo_imagefolder()

