import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import time
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
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4)

    return test_dataloader


def eval_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_dataloader = test_data_process()

    # 保存每轮准确率
    test_acc_list = []

    # 定义总数据量,预测正确样本个数,训练时间
    test_total_samples = 0
    test_total_correct = 0
    test_start = time.time()

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            model.eval()
            output = model(x)

            # 统计每批预测正确的样本数
            test_total_correct += (torch.argmax(output, dim=-1) == y).sum()
            # 统计每批的样本
            test_total_samples += len(y)

        print(f"epoch:{1}, val_acc:{test_total_correct / test_total_samples:.2f},"
              f" time:{time.time() - test_start:.3f}")
        print("-" * 30)

        # 记录准确率
        test_acc_list.append(round(test_total_correct.item() / test_total_samples, 2))

    test_process = pd.DataFrame({
        "epoch": 1,
        "test_acc": test_acc_list})

    # 保存测试过程数据到CSV文件
    with open("./test_process.csv", "w", encoding="utf-8") as f:
        test_process.to_csv(f, index=False)


if __name__ == '__main__':
    model = LeNet()
    model.load_state_dict(torch.load("./best_LeNet_model.pth"))
    # eval_model(model)

    test_dataloader = test_data_process()
    """
    classes
    ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
     'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    """
    classes_list = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                    "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    with torch.no_grad():
        count = 0
        for x, y in test_dataloader:
            if count >= 50:
                break
            count += 1
            x = x.to("cuda")
            y = y.to("cuda")
            model.to("cuda")
            model.eval()
            output = model(x)
            y_label = y.item()
            pre_label = torch.argmax(output, dim=-1).item()
            print(f"真实值: {classes_list[y_label]}<------->预测值: {classes_list[pre_label]}")

