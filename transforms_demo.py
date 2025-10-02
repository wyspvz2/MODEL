from torchvision import transforms

# 定义图像预处理流程
transform = transforms.Compose([
    transforms.ToTensor(),                      # 将 PIL Image 或 numpy 转为 tensor，值范围 [0,1]
    transforms.Normalize((0.5,), (0.5,))       # 标准化：均值0.5，标准差0.5 -> 范围 [-1,1]
])
