# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 08:54
# @Author  : AaronJny
# @File    : create_classify_dataset.py
# @Desc    : 创建二分类器的数据集
import json
import random
from glob import glob
from os.path import abspath


def create_dataset():
    correct_word_images = glob('correct_words/*.png')
    # 建立原始字（即标签部分的字）->图片路径列表的映射
    correct_word_map = {}
    for image in correct_word_images:
        word = image.split('/')[-1].split('-')[0]
        # 过滤掉认不出来的字
        if word == '1':
            continue
        correct_word_map.setdefault(word, []).append(image)
    # 建立生成字（即验证部分的字）->图片路径列表的映射
    gen_word_images = glob('gen_words/*.png')
    gen_word_images_set = set(gen_word_images)
    gen_word_map = {}
    for image in gen_word_images:
        word = image.split('/')[-1].split('-')[0]
        # 过滤掉认不出来的字
        if word == '1':
            continue
        gen_word_map.setdefault(word, []).append(image)
    # 通过负采样生成数据集
    records = []
    for word, correct_images in correct_word_map.items():
        for correct_image in correct_images:
            gen_images = gen_word_map.get(word, [])
            tmp_records = []
            # 如果有生成的图片
            if gen_images:
                # 负采样数和正样本数相同
                sample_num = len(gen_images)
                # 添加正样本
                for gen_image in gen_images:
                    tmp_records.append((1, abspath(correct_image), abspath(gen_image)))
                # 随机选择负样本
                random_images = gen_word_images_set - set(gen_images)
                for image in random.sample(list(random_images), sample_num):
                    tmp_records.append((0, abspath(correct_image), abspath(image)))
            records.append(tmp_records)
    with open('data.json', 'w') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    create_dataset()
