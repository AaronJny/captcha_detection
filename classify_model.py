# -*- coding: utf-8 -*-
# @Time    : 2020/11/13
# @Author  : AaronJny
# @File    : classify_model.py
# @Desc    :
import tensorflow as tf
from tensorflow.keras import layers

from config import ClassifyConfig


class SiameseNetwork(tf.keras.Model):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.net_stage_1 = tf.keras.Sequential([
            layers.Input(shape=(*ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS)),
            layers.Conv2D(6, (3, 3), padding='same'),
            layers.MaxPooling2D((2, 2), 2),
            layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU(),
            layers.Conv2D(16, (5, 5)),
            layers.MaxPooling2D((2, 2), 2),
            layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU()
        ])
        self.net_stage_2 = tf.keras.Sequential([
            layers.Conv2D(6, (3, 3)),
            layers.MaxPooling2D((2, 2), 2),
            layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(84),
            # layers.Dropout(ClassifyConfig.DROPOUT_RATE),
            layers.ReLU(),
            layers.Dense(1, activation='sigmoid')
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        outs = []
        for x in inputs:
            out = self.net_stage_1(x)
            outs.append(out)
        out = tf.concat(outs, axis=-1)
        out = self.net_stage_2(out)
        return out


def load_classify_model():
    model = SiameseNetwork()
    model.build(
        [(None, *ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS),
         (None, *ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS)])
    model.load_weights(ClassifyConfig.MODEL_PATH)
    return model
