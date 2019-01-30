import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from os.path import join, basename
import argparse
from dataloader import DatasetFromFolder


# コマンドライン引数の受け取り
parser = argparse.ArgumentParser(description='a fork of pytorch pix2pix')
parser.add_argument('--model', type=str, default='trained_classifier.pth', help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
option = parser.parse_args()

# GPU使うって言ってるのに使えなかったら怒る
if option.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

labels = np.array(['agiri', 'botsu', 'others', 'sonya', 'yasuna', 'yasuna_sonya'])
label_indices = {label: index for index, label in enumerate(labels)}

test_set = DatasetFromFolder(join('dataset', 'test'), label_indices, 'test')
test_data_loader = DataLoader(dataset=test_set, batch_size=1)

classifier = torch.load(option.model)
print(classifier)


def test():
    classifier.eval()  # 評価モードに変更
    with torch.no_grad():  # 無駄な勾配計算を行わないようにする
        all_correct_num = 0
        correct_num = [0] * 6
        label_num = [0] * 6
        # テストデータを1つずつ読み込み
        for iteration, (image, label, image_path) in enumerate(test_data_loader, 1):
            # Variableで包む
            if option.cuda:
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)
            predicted = classifier.forward(image)
            predicted_label = torch.argmax(predicted)

            label_num[label] += 1
            if predicted_label == label:
                all_correct_num += 1
                correct_num[label] += 1
            predicted = predicted.cpu().numpy()[0]
            print('{}: {}'.format(basename(image_path[0]), labels[label.data]))
            indices = np.argsort(predicted)[::-1]
            print('o ' if predicted_label == label else 'x ', end='')
            for index in indices:
                print('{}: {:.04f} '.format(labels[index], predicted[index]), end='')
            print()

        print('accuracy: {:.04f}'.format(all_correct_num / len(test_data_loader)))
        for i in range(6):
            print('{}: {:.04f}'.format(labels[i], correct_num[i] / label_num[i]))
        print()


if __name__ == '__main__':
    test()
