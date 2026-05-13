from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
# 定义训练数据的预处理流程：转换为Tensor并调整图像尺寸为224x224
train_compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224)
])

# 加载FashionMNIST训练数据集，自动下载到./data目录
train_data = FashionMNIST(root="./data",
                          train=True,
                          transform=train_compose,
                          download=True)

# 创建数据加载器，设置批量大小为64，并随机打乱数据
train_loader = data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True)

'''
# 获取一个批次的数据用于可视化或检查
for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维,并转成numpy数组
print(batch_x.shape)
batch_y = b_y.numpy()  # 将张量转成numpy数组
print(batch_y.shape)
class_label = train_data.classes  # 训练集的标签
print(class_label)

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()
'''