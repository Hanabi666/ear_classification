# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:05:39 2020

@author: DELL
"""

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import numpy as np


def read_label(root, is_train, is_test):
    if is_train:
        label_path = root + '/ImageSets/Main/train.txt'
    elif is_test:
        label_path = None
        return None
    else:
        label_path = root + '/ImageSets/Main/val.txt'
    with open(label_path, 'r', encoding='utf-8') as f:
        label_list = [x.strip().rstrip('\n') for x in f]
    return label_list


def padding(img):
    w, h = img.size
    img = np.array(img)
    if w > h:
        padh = (w - h)//2
        img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
    elif h > w:
        padw = (h - w)//2
        img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img

def read_diffclabel(path):
    with open(path + '/difficult_train.txt', 'r') as f:
        diffc_label = f.read().split('\n')
    diffc_label.pop(-1)
    diffc_label = [int(x) for x in diffc_label]
    return diffc_label


class EarDataset(data.Dataset):
    def __init__(self, root, transform=None, is_train=True, is_test=False):
        label_list = read_label(root, is_train, is_test)
        self.label_list = label_list
        self.data_list = [x.replace('.txt', '.jpg') for x in label_list]
        
        if is_train:
            self.label_path = root + '/Annotations/train/'
            self.data_path = root + '/JPEGImages/train/'
        else:
            self.label_path = root + '/Annotations/test/'
            self.data_path = root + '/JPEGImages/test/'

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        with open(self.label_path + self.label_list[index], 'r') as f:
            label = f.read().strip().split(' ')
        label = [float(x) for x in label]
        label = np.array(label)
        label = torch.from_numpy(label)
        data = Image.open(self.data_path + self.data_list[index])
        data = padding(data)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.label_list)


class dif_Dataset(data.Dataset):
    def __init__(self, root, dataset):
        self.dif_index = read_diffclabel(root)
        self.dataset = dataset

    def __getitem__(self, index):
        idx = self.dif_index[index]
        data, label = self.dataset[idx]
        return data, label

    def __len__(self):
        return len(self.dif_index)


if __name__ == '__main__':
    root = 'F:/ear_classifier_demo_1/data'
    
    train_dataset = EarDataset(root, None, True, False)
    dif_train_dataset = dif_Dataset(root, train_dataset)
    print(len(dif_train_dataset))
    data1, label1 = train_dataset[1873]
    data2, label2 = dif_train_dataset[299]
    print(data1==data2)
