# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:05:07 2020

@author: DELL
"""

import os
from PIL import Image


def read_label(label_path):
    with open(label_path, 'r') as f:
        label = f.read()
    label = label.rstrip('\n').split(' ')
    (clas, x, y, w, h) = [float(i) for i in label]
    return (x, y, w, h)


def trans(location):
    cx, cy, w, h = location
    x0 = cx - w/2
    y0 = cy - h/2
    x1 = cx + w/2
    y1 = cy + h/2
    return (x0, y0, x1, y1)


label_list = ['Annotations - 副本/' + x for x in os.listdir('Annotations - 副本/')]
label_list_ = [x for x in os.listdir('Annotations - 副本/')]
label = {}
for idx, i in enumerate(label_list):
    label[label_list_[idx].rstrip('.txt')] = read_label(i)

img_list = [x.rstrip('.txt') for x in os.listdir('Annotations - 副本/')]

for i in img_list:
    img = Image.open('JPEGImages - 副本/' + i + '.jpg')
    name = label[i]
    x0, y0, x1, y1 = trans(name)
    w, h = img.size
    x0, y0, x1, y1 = x0*w, y0*h, x1*w, y1*h
    img = img.crop((x0, y0, x1, y1))
    img.save('jpg/'+i+'.jpg', quality=100)


















