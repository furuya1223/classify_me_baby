import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()  # おまじない（親クラスのコンストラクタを明示的に呼ばないといけない）
        # モデル定義
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)   # 畳み込み層
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)   # 畳み込み層
        self.fc1 = nn.Linear(7*7*64, 1024)     # 全結合層
        self.fc2 = nn.Linear(1024, 6)         # 全結合層

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(x, 2, 2)  # 縦横ともに半分になる（28->14）
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, 2, 2)  # 縦横ともに半分になる（14->7）
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
