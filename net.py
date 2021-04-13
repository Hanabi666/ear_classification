import torch
import torch.nn as nn
from mobilenetv3 import *
from mobilenetv3 import h_swish


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = mobilenetv3_large()
        self.net.load_state_dict(torch.load('F:/ear_classifier_demo/checkpoints/mobilenetv3-large-1cd25616.pth'))
        self.net.classifier = nn.Sequential(
        nn.Linear(self.net.classifier[0].in_features, 1280, bias=True),
        h_swish(),
        nn.Dropout(0.2),
        nn.Linear(1280, 500, bias=True),
        h_swish(),
        nn.Dropout(0.2),
        nn.Linear(500, 10),
        nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == '__main__':
    net = Net()
    print(net)
