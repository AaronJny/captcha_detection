# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 21:46
# @Author  : AaronJny
# @File    : convert.py
# @Desc    :
# 引入转换脚本
from xyolo import voc2xyolo

# voc格式的标注数据路径的正则表达式
input_path = './labels/*.xml'
# classes是我们要检测的所有有效类别名称构成的txt文件，每个类别一行
classes_path = './classes.txt'
# 转换后的xyolo数据集存放路径
output_path = './xyolo_label.txt'
# 开始转换
voc2xyolo(input_path=input_path, classes_path=classes_path, output_path=output_path)
