# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 22:28
# @Author  : AaronJny
# @File    : test_detect.py
# @Desc    :
from xyolo import YOLO

from config import XYoloConfig

yolo = YOLO(XYoloConfig())
img = yolo.detect_and_draw_image('./images/286.png')
img.show()
