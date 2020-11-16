# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 22:18
# @Author  : AaronJny
# @File    : config.py
# @Desc    :
from xyolo import DefaultYolo3Config


# 创建一个DefaultYolo3Config的子类，在子类里覆盖默认的配置
class XYoloConfig(DefaultYolo3Config):
    def __init__(self):
        super(XYoloConfig, self).__init__()
        # 数据集路径，推荐使用绝对路径
        self._dataset_path = '/Users/aaron/code/captcha_detection/xyolo_label.txt'
        # 类别名称文件路径，推荐使用绝对路径
        self._classes_path = '/Users/aaron/code/captcha_detection/classes.txt'
        # 模型保存路径，默认是保存在当前路径下的xyolo_data下的，也可以进行更改
        # 推荐使用绝对路径
        self._output_model_path = 'detect_model.h5'


class ClassifyConfig:
    # 数据集路径
    DATA_PATH = './data.json'
    # 验证集比例
    VAL_SPLIT = 0.2
    # 测试集比例
    TEST_SPLIT = 0
    # batch_size
    BATCH_SIZE = 32
    # Dropout
    DROPOUT_RATE = 0.2
    # imagenet数据集均值
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    # imagenet数据集标准差
    IMAGE_STD = [0.299, 0.224, 0.225]
    # 学习率
    LR = 1e-3
    # 图片大小
    IMAGE_SIZE = (32, 32)
    # 图像信道
    IMAGE_CHANNELS = 1
    # 模型保存地址
    MODEL_PATH = './word_classify_best_weights.h5'
