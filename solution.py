# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 16:27
# @Author  : AaronJny
# @File    : solution.py
# @Desc    : 点选式验证码解决方案
import random
from glob import glob

import cv2
import numpy as np
from PIL import Image
from xyolo import YOLO

from classify_model import load_classify_model
from config import XYoloConfig
from process_classify_dataset import preprocess_image

config = XYoloConfig()
yolo = YOLO(config)
siamese_network = load_classify_model()


def detect(image_path):
    # 接收图片
    img = Image.open(image_path)
    img_array = np.asarray(img.convert('RGB'))
    # 提取两个标签
    word1_array = img_array[1:34, 186:216]
    word1_image = Image.fromarray(word1_array)
    word2_array = img_array[1:34, 216:246]
    word2_image = Image.fromarray(word2_array)
    # 检测文字
    results = yolo.detect_image(img)
    word_images = [word1_image, word2_image]
    word_arrays = [preprocess_image(x) for x in word_images]
    word_arrays = [x.reshape((1, *x.shape)) for x in word_arrays]
    correct_pos = []
    scores = []
    for word_array in word_arrays:
        sub_scores = []
        for index, (_, _, _, x1, y1, x2, y2) in enumerate(results):
            tmp_img_array = img_array[y1:y2, x1:x2]
            tmp_img_array = preprocess_image(Image.fromarray(tmp_img_array))
            tmp_img_array = tmp_img_array.reshape((1, *tmp_img_array.shape))
            score = siamese_network.predict([word_array, tmp_img_array])[0]
            sub_scores.append((score, index))
            print(score)
        sub_scores.sort(key=lambda x: -x[0])
        scores.append(sub_scores)
    score1, index1 = scores[0][0]
    score2, index2 = scores[1][0]
    if index1 == index2:
        if score1 > score2:
            index2 = scores[1][1][1]
        else:
            index1 = scores[0][1][1]
    for index in (index1, index2):
        correct_pos.append(results[index][3:])
    colors = [(0, 255, 0), (0, 0, 255)]
    for (x1, y1, x2, y2), color in zip(correct_pos, colors):
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
    cv2.imshow('result', img_array)
    for pos in correct_pos:
        print(pos)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


def run():
    images = glob('./images/2??.png')
    correct_cnt = 0
    half_correct_cnt = 0
    total = 30
    for image_path in random.sample(images, total):
        key = detect(image_path)
        # 如果正确选中了标签中的两个字，就按数字1
        # 注意，所谓的正确是指顺序也要正确。
        # 第一个字用绿框圈起来，第二个字用红框
        if key == ord('1'):  # 数字1
            correct_cnt += 1
            half_correct_cnt += 1
        # 如果只正确选中了其中一个字，就按数字2
        # 注意，这里的正确也包括了顺序，顺序不对不认为是正确的
        elif key == ord('2'):  # 数字2
            half_correct_cnt += 1
        # 都不是就按任意键
        else:
            pass
    print('识别通过率: {}'.format(correct_cnt / total))
    print('至少识别出一个的概率: {}'.format(half_correct_cnt / total))


if __name__ == '__main__':
    run()
