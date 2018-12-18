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
        self.conv1_1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)   # 畳み込み層
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)   # 畳み込み層
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)   # 畳み込み層
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)   # 畳み込み層
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)   # 畳み込み層
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)   # 畳み込み層
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)   # 畳み込み層
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)   # 畳み込み層
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)   # 畳み込み層
        self.drop1 = nn.Dropout2d()
        self.fc1 = nn.Linear(16*16*128, 500)     # 全結合層
        self.drop2 = nn.Dropout()
        self.fc2 = nn.Linear(500, 6)         # 全結合層

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = F.max_pool2d(x, 2, 2)  # 縦横ともに半分になる（128->64）
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.max_pool2d(x, 2, 2)  # 縦横ともに半分になる（64->32）
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(self.drop1(x), 2, 2)  # 縦横ともに半分になる（32->16）
        x = x.view(-1, 16*16*128)
        x = self.drop2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
