# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 09:22
# @Author  : AaronJny
# @File    : train_classify.py
# @Desc    :
from tensorflow.keras import losses, optimizers, callbacks

from classify_model import SiameseNetwork
from config import ClassifyConfig
from process_classify_dataset import load_dataset

model = SiameseNetwork()
model.build(
    [(None, *ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS),
     (None, *ClassifyConfig.IMAGE_SIZE, ClassifyConfig.IMAGE_CHANNELS)])
model.summary()
model.compile(optimizer=optimizers.Adam(ClassifyConfig.LR), loss=losses.binary_crossentropy, metrics=['accuracy'])

train_data, val_data, test_data = load_dataset()

checkpoint = callbacks.ModelCheckpoint(ClassifyConfig.MODEL_PATH,
                                       monitor='val_accuracy', save_best_only=True, save_weights_only=True)
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1)
model.fit(train_data.for_fit(), epochs=300, validation_data=val_data.for_fit(), steps_per_epoch=train_data.steps,
          validation_steps=val_data.steps, callbacks=[checkpoint])

if ClassifyConfig.TEST_SPLIT:
    print(model.evaluate(test_data.for_fit(), steps=test_data.steps))
