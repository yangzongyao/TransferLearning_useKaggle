#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 01:12:20 2018

@author: yang
"""

import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def processRGB(img):
    t = []
    for i in img:
        R = i[0:1024].reshape(32,32)
        G = i[1024:2048].reshape(32,32)
        B = i[2048:3072].reshape(32,32)
        image = np.dstack((R,G,B))
        t.append(image)
    return np.array(t)


def CIFAR10_getData():
    train_data = []
    train_label = []
    for i in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','batches.meta']:
        dic = unpickle('cifar-10-batches-py/'+i)
        for i in dic.keys():
            if i == b'data':
                train_data = [*train_data,*dic[i]]
                train_data = np.array(train_data)
            elif i == b'labels':
                train_label = [*train_label, *dic[i]]
                train_label = np.array(train_label)
            elif i == b'label_names':
                label_name = dic[i]

    dic_test = unpickle('cifar-10-batches-py/test_batch')
    test_data = dic_test[b'data']
    test_label = np.array(dic_test[b'labels'])
    train_data = processRGB(train_data)
    test_data = processRGB(test_data)
    return train_data, train_label, test_data, test_label, label_name
    
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    return img
    
