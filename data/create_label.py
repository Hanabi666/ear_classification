# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:36:07 2020

@author: DELL
"""
# [1351, 67, 233, 91, 246, 656, 120, 117, 36, 66]

import os
import random

root = 'F:/ear_classifier_demo_1/data/JPEGImages/train'
imgs = os.listdir(root)

i = 0
num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for img in imgs:
    i += 1
    num[0]+=1
    array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    if '招风耳' in img:
        array[0] = 1
        num[1]+=1
    if '垂耳' in img:
        array[1] = 1
        num[2]+=1
    if '杯状耳' in img:
        array[2] = 1
        num[3]+=1
    if '狒狒耳' in img:
        array[3] = 1
        num[4]+=1
    if '耳轮畸形' in img:
        array[4] = 1
        num[5]+=1
    if '耳轮脚横突' in img:
        array[5] = 1
        num[6]+=1
    if '环缩耳' in img:
        array[6] = 1
        num[7]+=1
    if '隐耳' in img:
        array[7] = 1
        num[8]+=1
    if '耳甲腔狭窄' in img:
        array[8] = 1
        num[9]+=1
    #with open('F:/ear_classifier_demo_1/data/Annotations/train/'+img.replace('.jpg', '.txt'), 'w') as f:
    #    f.writelines([str(x)+' ' for x in array])

    print('{}/1351'.format(i))
print(num)


#with open('F:/ear_classifier_demo_1/data/ImageSets/Main/train.txt', 'w', encoding='utf-8') as f:
 #   f.writelines([str(x).replace('.jpg', '.txt')+'\n' for x in imgs])
"""
with open('F:/ear_classifier_demo_1/data/ImageSets/Main/val.txt', 'w', encoding='utf-8') as f:
    f.writelines([str(x).replace('.jpg', '.txt')+'\n' for x in imgs[int(0.8*len(imgs)):]])
"""





