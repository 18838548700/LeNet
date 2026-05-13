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


def test_data_process():
    # 加载FashionMNIST训练数据集，自动下载到./data目录
    test_data = FashionMNIST(root="./data",
                             train=False,
                             transform=train_compose,
                             download=True)

    # 创建训练集数据加载器
    test_dataloader = data.DataLoader(dataset=test_data,
                                      batch_size=64,
                                      shuffle=False,
                                      num_workers=4)

    return test_dataloader


def eval_model(model, epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_dataloader = test_data_process()


    # 保存每轮的训练和验证损失, 准确率
    test_lost_list = []
    test_acc_list = []

    for epoch in range(epochs):
        # 定义变量,总损失,总数据量,预测正确样本个数,训练时间
        test_total_loss = 0.0
        test_total_samples = 0
        test_total_correct = 0
        test_start = time.time()

        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            model.eval()

            output = model(x)
            loss = criterion(output, y)

            # 测试没有更新参数过程

            # 统计每批预测正确的样本数
            test_total_correct += (torch.argmax(output, dim=-1) == y).sum()
            # 统计每批的样本
            test_total_samples += len(y)
            # 统计每批的损失
            test_total_loss += loss.item() * len(y)

        print(f"epoch:{epoch + 1}, val_acc:{test_total_correct / test_total_samples:.2f},"
              f" time:{time.time() - test_start:.3f}")
        print("-" * 30)

        # 记录每轮的损失和准确率
        test_lost_list.append(test_total_loss)
        test_acc_list.append(round(test_total_correct.item() / test_total_samples, 2))

    test_process = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "test_lost": test_lost_list,
        "test_acc": test_acc_list})

    # 保存测试过程数据到CSV文件
    with open("./test_process.csv", "w", encoding="utf-8") as f:
        test_process.to_csv(f, index=False)

if __name__ == '__main__':
    model = LeNet()
    model.load_state_dict(torch.load("./best_LeNet_model.pth"))
    eval_model(model, epochs=1)