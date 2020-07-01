import glob
from enum import Enum

import pandas as pd

from util import utils


class Dataset(Enum):
    train = 'train'
    validation = 'validation'
    test = 'test'


def samples():
    source_path = '/home/lukasz/Studia/Analiza obrazow i wideo/ALPR/SimpleALPR/dataset/'
    train_directory_path = source_path + 'test_*.png'
    images = sorted(glob.glob(train_directory_path))
    for image in images:
        yield utils.load_image(image), image.split('/')[-1].split('.')[0]

def samples_v2():
    source_path = '/home/lukasz/Studia/Analiza obrazow i wideo/ALPR/SimpleALPR/dataset_v2/*/'
    train_directory_path = source_path + '*.jpg'
    images = sorted(glob.glob(train_directory_path))
    for image in images:
        yield utils.load_image(image), image.split('/')[-2] + image.split('/')[-1].split('.')[0]

def sample(number):
    source_path = '/home/lukasz/Studia/Analiza obrazow i wideo/ALPR/SimpleALPR/dataset/'
    sample_path = source_path + 'test_{}.jpg'.format(number)
    return utils.load_image(sample_path), sample_path.split('/')[-1].split('.')[0]


class DatasetsProvider:

    def __init__(self, source_path):
        self.path = source_path

    def load_train(self, img_ext='.png'):
        train_directory_path = self.path + 'training' + '/**/*'
        images = sorted(glob.glob(train_directory_path + img_ext))
        labels = sorted(glob.glob(train_directory_path + '.txt'))

        df = pd.DataFrame({'image': images, 'label': labels})
        return df

    def images(self):
        df = self.load_train()
        for index, row in df.iterrows():
            image = utils.load_image(row.image)
            label = self._label_file_to_dict(row.label)
            plate_position = label['position_plate'].strip()
            plate_number = label['plate'].strip()

            yield (image, plate_position, plate_number)

    def _label_file_to_dict(self, path):
        d = {}
        with open(path) as file:
            for line in file:
                (key, val) = line.split(':')
                d[key] = val
        return d


if __name__ == '__main__':
    # dp = DatasetsProvider(
    #     source_path='/home/lukasz/Studia/Analiza obrazow i wideo/UFPR-ALPR dataset/'
    # )
    #
    # for example in dp.images():
    #     print(example)


    for example in samples():
        print(example)
