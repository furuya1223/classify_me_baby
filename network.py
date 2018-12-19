import torch.nn as nn
import torch.nn.functional as F


# 分類器の定義
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()  # おまじない（親クラスのコンストラクタを明示的に呼ばないといけない）
        # モデル定義
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)   # 畳み込み層
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)  # 畳み込み層
        self.fc1 = nn.Linear(7*7*64, 1024)  # 全結合層
        self.fc2 = nn.Linear(1024, 6)       # 全結合層

    # 前向き計算
    def forward(self, x):
        # 入力 x のサイズは (batchSize, 3, 28, 28)
        x = F.relu(self.conv1(x))   # 畳み込みと活性化関数
        x = F.max_pool2d(x, 2, 2)   # 縦横ともに半分になる（28->14）
        x = F.relu(self.conv2(x))   # 畳み込みと活性化関数
        x = F.max_pool2d(x, 2, 2)   # 縦横ともに半分になる（14->7）
        x = x.view(-1, 7*7*64)      # 2階のテンソルにreshapeする（全結合層に入力するため）
        x = F.relu(self.fc1(x))     # 全結合層と活性化関数
        x = self.fc2(x)             # 全結合層
        return F.softmax(x, dim=1)  # softmaxに通して出力
