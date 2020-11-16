# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 22:08
# @Author  : AaronJny
# @File    : train_detect.py
# @Desc    :
# 导入包
from xyolo import YOLO
from xyolo import init_yolo_v3

from config import XYoloConfig

# 使用修改后的配置创建yolo对象
config = XYoloConfig()
init_yolo_v3(config)
# 如果是训练，在创建yolo对象时要传递参数train=True
yolo = YOLO(config, train=True)
# 开始训练，训练完成后会自动保存
yolo.fit()
