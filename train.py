import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from os.path import join
import argparse
import random

from dataloader import DatasetFromFolder
from network import Classifier


# コマンドライン引数の受け取り
parser = argparse.ArgumentParser(description='a fork of pytorch pix2pix')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=130, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
option = parser.parse_args()

# GPU使うって言ってるのに使えなかったら怒る
if option.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# 実行ごとに結果が変わらないように乱数シードを固定
random.seed(1)
torch.manual_seed(option.seed)
if option.cuda:
    torch.cuda.manual_seed(option.seed)
    cudnn.deterministic = True

# ラベルなどの準備
labels = np.array(['agiri', 'botsu', 'others', 'sonya', 'yasuna', 'yasuna_sonya'])
label_indices = {label: index for index, label in enumerate(labels)}
# weight = torch.Tensor([1/9, 1/2, 1/13, 1/35, 1/68, 1/9])

# データローダの用意
train_set = DatasetFromFolder(join('dataset', 'train'), label_indices, 'train')
train_data_loader = DataLoader(dataset=train_set, batch_size=option.batchSize, shuffle=True)
test_set = DatasetFromFolder(join('dataset', 'test'), label_indices, 'test')
test_data_loader = DataLoader(dataset=test_set, batch_size=1)

# 分類器と誤差関数の用意
if option.cuda:
    classifier = Classifier().cuda()
    # クロスエントロピー
    criterion = nn.CrossEntropyLoss().cuda()
else:
    classifier = Classifier()
    # クロスエントロピー
    criterion = nn.CrossEntropyLoss()

# Adamオプティマイザを用意
optimizer = optim.Adam(classifier.parameters(), lr=option.lr)

# ネットワーク構造の表示
print(classifier)


def train():
    for epoch in range(1, option.nEpochs + 1):
        print('Epoch: {}'.format(epoch))

        # 学習
        classifier.train()  # 学習モードに変更
        loss_total = 0

        # イテレーション
        for iteration, (image, label, _) in enumerate(train_data_loader, 1):
            # 読み込んだ画像データと正解ラベルをVariableで包む
            if option.cuda:
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)

            optimizer.zero_grad()  # オプティマイザが保持する勾配を初期化
            predicted = classifier.forward(image)  # 前向き計算（ラベル予測）
            loss = criterion(predicted, label)  # クロスエントロピー誤差の計算
            loss.backward()  # 後ろ向き計算（誤差逆伝播で勾配計算）
            optimizer.step()  # オプティマイザでパラメータを修正
            loss_total += loss.data

        # 今エポックのロスの平均値を出力
        print('Average Loss: {:.04f}'.format(loss_total / option.batchSize))

        # テスト
        classifier.eval()  # 評価モードに変更
        with torch.no_grad():  # 無駄な勾配計算を行わないようにする
            all_correct_num = 0
            correct_num = [0] * 6
            label_num = [0] * 6

            # テストデータのイテレーションを回す
            for iteration, (image, label, _) in enumerate(test_data_loader, 1):
                # Variableで包む
                if option.cuda:
                    image = Variable(image.cuda())
                    label = Variable(label.cuda())
                else:
                    image = Variable(image)
                    label = Variable(label)

                predicted = classifier.forward(image)  # 前向き計算（ラベル予測）
                predicted_label = torch.argmax(predicted)  # 確率が最大となるラベルを予測ラベルとする

                # ラベルごとのデータ数・正解数と全体の正解数を記録
                label_num[label] += 1
                if predicted_label == label:
                    all_correct_num += 1
                    correct_num[label] += 1

            # 全体の正解率を表示
            print('accuracy: {:.04f}'.format(all_correct_num / len(test_data_loader)))
            # ラベル別の正解率を表示
            for i in range(6):
                print('{:.04f}, '.format(correct_num[i] / label_num[i]), end='')
            print()

    # 学習済みモデルを保存
    torch.save({'state_dict': classifier.state_dict()}, 'checkpoint.pth')


if __name__ == '__main__':
    train()
