import torch
import numpy as np
import json
from torchvision import transforms
from PIL import Image


def padding(img):
    w, h = img.size
    img = np.array(img)
    if w > h:
        padh = (w - h)//2
        img = np.pad(img, ((padh, padh), (0, 0), (0, 0)),
                     'constant', constant_values=0)
    elif h > w:
        padw = (h - w)//2
        img = np.pad(img, ((0, 0), (padw, padw), (0, 0)),
                     'constant', constant_values=0)
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


def pil_to_tensor(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img


def list_to_json(li):
    clas = {'人耳': '{}%'.format(round(li[9]*100, 2)), '招风耳': '{}%'.format(round(li[0]*100, 2)),
            '垂耳': '{}%'.format(round(li[1]*100, 2)), '杯状耳': '{}%'.format(round(li[2]*100, 2)),
            '狒狒耳': '{}%'.format(round(li[3]*100, 2)), '耳轮畸形': '{}%'.format(round(li[4]*100, 2)),
            '耳轮脚横突': '{}%'.format(round(li[5]*100, 2)), '环缩耳': '{}%'.format(round(li[6]*100, 2)),
            '隐耳': '{}%'.format(round(li[7]*100, 2)), '耳甲腔狭窄': '{}%'.format(round(li[8]*100, 2))}
    Json = json.dumps(clas, ensure_ascii=False)
    return Json


