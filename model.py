import torch
import torch.nn as nn
from torchsummary import summary

"""
标准的 LeNet-5 结构应该是: 2层卷积, 3层全连接
input: 1x28×28 (CHW)
C1: Conv(1→6, 5×5) + Sigmoid + AvgPool(2×2)   6x14x14
C2: Conv(6→16, 5×5) + Sigmoid + AvgPool(2×2)  16x5x5
展平: Flatten()                                400
F1: Linear(400→120) + Sigmoid
F2: Linear(120→84) + Sigmoid
F3: Linear(84→10) (输出10个类别)
"""


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

'''
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    summary(model, (1, 28, 28))
'''

