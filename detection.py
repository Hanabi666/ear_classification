# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:00:13 2020

@author: DELL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data.dataset import *
from net import *
from shufflenet import *
from utils import *

train_data = EarDataset(ROOT, None, True, False)
val_data = EarDataset(ROOT, None, False, False)
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False,
                              num_workers=0, drop_last=False)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False,
                              num_workers=0, drop_last=True)


output_ = []
target_ = []

net = ShuffleNet()
net.load_state_dict(torch.load('F:/ear_classifier_demo_1/checkpoints/trained_resnet34_Batch16_20_epoch33.pth'))
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

for i in range(len(output_)):
    output_[i] = [round(x, 3) for x in output_[i]]
for i in range(len(output_)):
    print('{}\t{}\n'.format(output_[i], target_[i]))
