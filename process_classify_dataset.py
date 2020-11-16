# -*- coding: utf-8 -*-
# @Time    : 2020/11/13
# @Author  : AaronJny
# @File    : process_classify_dataset.py
# @Desc    :
import json
import math

import numpy as np
from PIL import Image

from config import ClassifyConfig


def preprocess_image(image: Image.Image):
    """
    对图片进行预处理，并返回处理后的图片矩阵
    """
    if isinstance(image, str):
        image = Image.open(image)
    x = image.resize(ClassifyConfig.IMAGE_SIZE)
    convert_flag = {
        1: 'L',
        3: 'RGB'
    }
    x = np.asarray(x.convert(convert_flag[ClassifyConfig.IMAGE_CHANNELS]), dtype=np.float32)
    x = x / 255.
    x = x.reshape((*ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS))
    return x


class DataGenerator:
    """
    封装了数据集的生成器
    """

    def __init__(self, data, batch_size=ClassifyConfig.BATCH_SIZE, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = int(math.ceil(len(self.data) / self.batch_size))

    @classmethod
    def load_image(cls, image_path):
        im = Image.open(image_path)
        image = np.asarray(im.convert("RGB"))
        del im
        return image

    @classmethod
    def preprocess_record(cls, record):
        """
        对一条给定数据进行预处理
        """
        label, image_path1, image_path2 = record
        image1 = preprocess_image(image_path1)
        image2 = preprocess_image(image_path2)
        return image1, image2, label

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data)
        total = len(self.data)
        for start in range(0, total, self.batch_size):
            end = min(total, start + self.batch_size)
            batch_images1 = []
            batch_images2 = []
            batch_labels = []
            for record in self.data[start:end]:
                image1, image2, label = self.preprocess_record(record)
                batch_images1.append(image1)
                batch_images2.append(image2)
                batch_labels.append(label)
            np_batch_images1 = np.asarray(batch_images1)
            np_batch_images2 = np.asarray(batch_images2)
            np_batch_labels = np.asarray(batch_labels)
            np_batch_labels.reshape((-1, 1))
            del batch_images1, batch_images2, batch_labels
            yield [np_batch_images1, np_batch_images2], np_batch_labels
            del np_batch_images1, np_batch_images2, np_batch_labels

    def for_fit(self):
        while True:
            yield from self.__iter__()


def load_dataset():
    """
    加载并划分、封装数据集，返回(训练数据集,验证数据集,测试数据集)
    """
    with open(ClassifyConfig.DATA_PATH, 'r') as f:
        records = json.load(f)
    np.random.shuffle(records)
    total = len(records)
    val_num = int(total * ClassifyConfig.VAL_SPLIT)
    test_num = int(total * ClassifyConfig.TEST_SPLIT)
    val_data = DataGenerator([x for record in records[:val_num] for x in record])
    test_data = DataGenerator([x for record in records[val_num:val_num + test_num] for x in record])
    train_data = DataGenerator([x for record in records[val_num + test_num:] for x in record])
    return train_data, val_data, test_data
