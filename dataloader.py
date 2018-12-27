import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import join
from os import listdir
from PIL import Image
import numpy as np


# データローダの定義（データを1つずつ読み込むもの）
class DatasetFromFolder(data.Dataset):
    # 初期化メソッド
    def __init__(self, image_dir, label_indices, mode='train'):
        super(DatasetFromFolder, self).__init__()

        # ディレクトリに存在するサブディレクトリとその中のファイルを列挙して
        # パスとラベルをリスト化（ラベルはサブディレクトリ名）
        # self.images = [(join(image_dir, label, filename), label)
        #                for label in listdir(image_dir)
        #                for filename in listdir(join(image_dir, label))]
        self.images = []
        for label in listdir(image_dir):
            label_images = [(join(image_dir, label, filename), label)
                            for filename in listdir(join(image_dir, label))]
            index = np.mod(range(273), len(label_images))
            self.images += label_images[index]

        self.label_indices = label_indices  # ラベル名から番号を得るための辞書

        # 学習用の前処理
        transform_train = [transforms.RandomRotation(30),      # -30度から+30度まででランダムに回転
                           transforms.Resize(28),              # 28x28 にリサイズ
                           transforms.RandomHorizontalFlip(),  # ランダムに左右反転
                           transforms.ToTensor(),              # Tensorに変換
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # [-1, 1] になるように変換
        # テスト用の前処理
        transform_test = [transforms.Resize(28),               # 28x28 にリサイズ
                          transforms.ToTensor(),               # Tensorに変換
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # [-1, 1] になるように変換

        # モードによって前処理を切り替え
        if mode == 'train':
            self.transform = transforms.Compose(transform_train)
        else:
            self.transform = transforms.Compose(transform_test)

    # indexを指定してデータを取得するメソッド
    def __getitem__(self, index):
        # 画像のパスとラベルを取得
        image_path, label = self.images[index]
        # 画像読み込み
        image = Image.open(image_path).convert('RGB')
        # 前処理
        image = self.transform(image)

        # 画像データ（Tensor），ラベル番号，画像パスを出力
        return image, self.label_indices[label], image_path

    # データセットのサイズを取得するメソッド
    def __len__(self):
        return len(self.images)
