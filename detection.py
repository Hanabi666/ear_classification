# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:00:13 2020

@author: DELL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Resnet34
from PIL import Image
from utils import *


net = Resnet34()
net.load_state_dict(torch.load('../classification_demo/checkpoints/trainedmodel.pth'))

def predict(img):
    img = padding(img)
    img = pil_to_tensor(img)
    print(img.size())
    img = torch.unsqueeze(img, dim=0)
    net.eval()
    with torch.no_grad():
        output = net(img)
    return (output.numpy().tolist())[0]

"""
optimizer = optim.SGD(net.parameters(), 0.05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
criterion = nn.BCELoss()
net.eval()
for batch_idx, (data, target) in enumerate(train_dataloader):
    data, target = data.to(DEVICE), target.float().to(DEVICE)
    with torch.no_grad():
        output = net(data)
        loss = criterion(output, target)
        #loss_.append(loss.item())

        output_.append((output.numpy().tolist())[0])
        target_.append((target.numpy().tolist())[0])
    print('{}/{}'.format(batch_idx+1, len(train_dataloader)))
"""

if __name__ == '__main__':
    root = 'classification_demo/test/正常.png'
    img = Image.open(root)
    output = predict(img)
    output = list_to_json(output)
    print(output)




