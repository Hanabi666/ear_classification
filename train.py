# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 10:15:12 2020

@author: DELL
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data.dataset import *
from net import *
from shufflenet import *
from utils import *


train_data = EarDataset(ROOT, None, True, False)
val_data = EarDataset(ROOT, None, False, False)
train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False,
                              num_workers=0, drop_last=True)
dif_train_dataset = dif_Dataset(ROOT, train_data)
dif_train_dataloader = DataLoader(dif_train_dataset, batch_size=32, shuffle=True,
                              num_workers=0, drop_last=True)

net = ShuffleNet()
#net.load_state_dict(torch.load('F:/ear_classifier_demo_1/checkpoints/trained_resnet34_Batch16_20_epoch33.pth'))
"""
optimizer = optim.SGD([
    {'params': net.net.layer3.parameters(), 'lr': 0.05},
    {'params': net.net.layer4.parameters(), 'lr': 0.05},
    {'params': net.net.fc.parameters(), 'lr': 0.05}
], lr=0.05)
"""
optimizer = optim.SGD(net.parameters(), 0.05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.BCELoss()


def train(model, device, train_dataloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader, 1):
        data, target = data.to(device), target.float().to(device) 
        output = model(data)

        # print(output)
        # print(output[:, 0])
        # print(target[:, 0])

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx) % 5 == 0:
            print('Train epoch:{} [{}/{} {:.0f}%]\tloss:{:.6f}\tlr:{}'.format(
                epoch, (batch_idx)*len(data), len(train_dataloader.dataset),
                100.*batch_idx/len(train_dataloader), loss.item(),
                optimizer.param_groups[0]['lr']))


def val(model, device, val_dataloader):
    loss = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(val_dataloader):
        data, target = data.to(device), target.float().to(device)
        with torch.no_grad():
            output = model(data)
            loss += criterion(output, target).item()
        #print(output)
    print('val_loss = {}'.format(loss/len(val_dataloader)))
    return loss/len(val_dataloader)


for epoch in range(EPOCHS):
    train(net, DEVICE, train_dataloader, optimizer, epoch)
    scheduler.step()
    val_loss = val(net, DEVICE, val_dataloader)
    torch.save(net.state_dict(), 'F:/ear_classifier_demo_1/checkpoints/dif_trained_resnet34_Batch4_20_epoch{}.pth'.format(epoch))















