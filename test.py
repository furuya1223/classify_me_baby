import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from os.path import join, basename
import argparse

from dataloader import DatasetFromFolder
from network import Classifier


# Training settings
parser = argparse.ArgumentParser(description='a fork of pytorch pix2pix')
parser.add_argument('--model', type=str, default='trained_classifier.pth', help='model file to use')
parser.add_argument('--image', type=str, default='', help='image to be classified')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
option = parser.parse_args()

if option.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

labels = np.array(['agiri', 'botsu', 'others', 'sonya', 'yasuna', 'yasuna_sonya'])
label_indices = {label: index for index, label in enumerate(labels)}

test_set = DatasetFromFolder(join('dataset', 'test'), label_indices, 'test')
test_data_loader = DataLoader(dataset=test_set, batch_size=1)

classifier = torch.load(option.model)
print(classifier)


def test():
    classifier.eval()
    with torch.no_grad():
        if option.image == '':
            all_correct_num = 0
            correct_num = [0] * 6
            label_num = [0] * 6
            for iteration, (image, label, image_path) in enumerate(test_data_loader, 1):
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
                predicted = predicted.cpu().numpy()
                print(basename(image_path))
                indices = np.argsort(predicted[0])[::-1]
                print(indices)
                for index in indices:
                    print('{}: {:.04f} '.format(label[index], predicted[index]), end='')
                print()

            print('accuracy: {:.04f}'.format(all_correct_num / len(test_data_loader)))
            for i in range(6):
                print('{:.04f}, '.format(correct_num[i] / label_num[i]), end='')
            print()
        else:
            pass


if __name__ == '__main__':
    test()
