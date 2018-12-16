import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import join
from os import listdir
from PIL import Image
import numpy as np


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, label_indices):
        super(DatasetFromFolder, self).__init__()
        self.images = [(join(image_dir, label, filename), label)
                       for label in listdir(image_dir)
                       for filename in listdir(join(image_dir, label))]
        self.label_indices = label_indices

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        image_path, label = self.images[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, self.label_indices[label]

    def __len__(self):
        return len(self.images)
