import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
import argparse

from dataloader import DatasetFromFolder
from network import Classifier


# Training settings
parser = argparse.ArgumentParser(description='a fork of pytorch pix2pix')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
option = parser.parse_args()

if option.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

labels = np.array(['agiri', 'botsu', 'others', 'sonya', 'yasuna', 'yasuna_sonya'])
label_indices = {label: index for index, label in enumerate(labels)}
weight = torch.Tensor([1/9, 1/2, 1/13, 1/35, 1/68, 1/9])

train_set = DatasetFromFolder(join('dataset', 'train'), label_indices, 'train')
train_data_loader = DataLoader(dataset=train_set, batch_size=option.batchSize, shuffle=True)
test_set = DatasetFromFolder(join('dataset', 'test'), label_indices, 'test')
test_data_loader = DataLoader(dataset=test_set, batch_size=1)


if option.cuda:
    classifier = Classifier().cuda()
    criterion = nn.CrossEntropyLoss(weight=weight.cuda()).cuda()
else:
    classifier = Classifier()
    criterion = nn.CrossEntropyLoss(weight=weight)

optimizer = optim.Adam(classifier.parameters())

print(classifier)


def train():
    for epoch in range(1, option.nEpochs + 1):
        print('Epoch: {}'.format(epoch))
        for iteration, (image, label) in enumerate(train_data_loader, 1):
            if option.cuda:
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)
            optimizer.zero_grad()
            predicted = classifier.forward(image)
            loss = criterion(predicted, label)
            loss.backward()
            optimizer.step()
            print('Iteration: {}/{}, Loss: {:.04f}'.format(iteration, len(train_data_loader), loss.data))

        # test
        with torch.no_grad():
            all_correct_num = 0
            correct_num = [0] * 6
            label_num = [0] * 6
            for iteration, (image, label) in enumerate(test_data_loader, 1):
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

            print('accuracy: {:.04f}'.format(all_correct_num / len(test_data_loader)))
            for i in range(6):
                print('{:.04f}, '.format(correct_num[i] / label_num[i]), end='')
            print()


if __name__ == '__main__':
    train()
