import copy
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import LeNet

train_compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(28)
])


def train_val_data_process():
    """
    处理并加载训练集和验证集数据
    
    该函数负责加载FashionMNIST数据集，并将其按8:2的比例随机划分为训练集和验证集。
    随后创建对应的DataLoader以支持批量数据加载和并行读取。
    
    Returns:
        tuple: 包含两个DataLoader对象
            - train_dataloader (DataLoader): 训练集数据加载器，batch_size=64，开启shuffle和多线程
            - val_dataloader (DataLoader): 验证集数据加载器，batch_size=64，关闭shuffle，开启多线程
    """
    # 加载FashionMNIST训练数据集，自动下载到./data目录
    train_data = FashionMNIST(root="./data",
                              train=True,
                              transform=train_compose,
                              download=True)

    # 按8:2比例随机划分训练集和验证集
    train_data, val_date = data.random_split(train_data,
                                             [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    # 创建训练集数据加载器
    train_dataloader = data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=4)
    # 创建验证集数据加载器
    val_dataloader = data.DataLoader(dataset=val_date,
                                     batch_size=64,
                                     shuffle=True,
                                     num_workers=4)
    return train_dataloader, val_dataloader


def train_model(model, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 记录当前时间
    now = time.time()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    train_dataloader, val_dataloader = train_val_data_process()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 保存每轮的训练和验证损失, 准确率
    train_lost_list = []
    train_acc_list = []
    val_lost_list = []
    val_acc_list = []

    for epoch in range(epochs):
        # 定义变量,总损失,总数据量,预测正确样本个数,训练时间
        train_total_loss = 0.0
        train_total_samples = 0
        train_total_correct = 0
        train_start = time.time()

        # 定义变量,总损失,总数据量,预测正确样本个数,训练时间
        val_total_loss = 0.0
        val_total_samples = 0
        val_total_correct = 0
        val_start = time.time()

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            model.train()

            output = model(x)
            loss = criterion(output, y)

            # 固定写法
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计每批预测正确的样本数
            train_total_correct += (torch.argmax(output, dim=-1) == y).sum()
            # 统计每批的样本
            train_total_samples += len(y)
            # 统计每批的损失
            train_total_loss += loss.item() * len(y)

        # 打印损失信息
        print(
            f"epoch:{epoch + 1}, loss:{train_total_loss / train_total_samples:.5f},"
            f" train_acc:{train_total_correct / train_total_samples:.2f},"
            f" time:{time.time() - train_start:.3f}")

        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            model.eval()

            output = model(x)
            loss = criterion(output, y)

            # 测试没有更新参数过程

            # 统计每批预测正确的样本数
            val_total_correct += (torch.argmax(output, dim=-1) == y).sum()
            # 统计每批的样本
            val_total_samples += len(y)
            # 统计每批的损失
            val_total_loss += loss.item() * len(y)

        print(f"epoch:{epoch + 1}, val_acc:{val_total_correct / val_total_samples:.2f},"
              f" time:{time.time() - val_start:.3f}")
        print("-" * 30)

        # 记录每轮的损失和准确率
        train_lost_list.append(round(train_total_loss / train_total_samples, 2))
        val_lost_list.append(round(val_total_loss / train_total_samples, 2))
        train_acc_list.append(round(train_total_correct.item() / train_total_samples, 2))
        val_acc_list.append(round(val_total_correct.item() / val_total_samples, 2))

        # 记录最高的准确率和当次的模型参数
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    # 保存准确率最高的模型
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "./best_LeNet_model.pth")

    train_process = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_lost": train_lost_list,
        "val_lost": val_lost_list,
        "train_acc": train_acc_list,
        "val_acc": val_acc_list})

    # 保存训练过程数据到CSV文件
    with open("./train_process.csv", "w", encoding="utf-8") as f:
        train_process.to_csv(f, index=False)

    # 记录训练总时间
    time_use = time.time() - now
    print("训练和验证耗费的时间: {:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    return train_process


def plot_process(train_process):
    """
    绘制训练过程中的损失和准确率变化曲线
    
    该函数接收训练过程数据，使用matplotlib创建包含两个子图的可视化图表：
    - 左侧子图显示训练损失随epoch的变化趋势
    - 右侧子图显示训练准确率随epoch的变化趋势
    
    Args:
        train_process (pd.DataFrame): 包含训练过程数据的DataFrame，必须包含以下列：
            - epoch: 训练轮次
            - train_lost: 每轮的训练损失值
            - train_acc: 每轮的训练准确率值
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制训练损失曲线
    ax[0].plot(train_process["epoch"], train_process["train_lost"], "ro-", label="train_lost")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Training Loss")
    ax[0].legend()

    # 绘制训练准确率曲线
    ax[1].plot(train_process["epoch"], train_process["train_acc"], "bo-", label="train_acc")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("acc")
    ax[1].set_title("Training Accuracy")
    ax[1].legend()

    plt.savefig("./train.svg")
    plt.savefig("./train.png")

    # 自动调整子图间距，防止标签、标题等元素重叠
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = LeNet()
    # train_process = train_model(model, epochs=20)
    train_process = pd.read_csv("./train_process.csv")
    plot_process(train_process)