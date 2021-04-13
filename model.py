import torch
import torch.nn as nn
import torchvision.models as models

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        #self.net = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        #self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=False)
        self.net = models.resnet34(pretrained=False)
        #self.net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.net.fc = nn.Sequential(
        nn.Linear(self.net.fc.in_features, 10, bias=True),
        #torch.sigmoid()
        )
        """
        self.classifier0 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier1 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier2 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier3 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier4 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier5 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier6 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier7 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier8 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self.classifier9 = nn.Sequential(
            #nn.Conv2d(1024, 512, 1, 1, bias=False),
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 100, bias=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        """
    def forward(self, x):
        x = self.net(x)
        """
        x0 = self.classifier0(x)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        x4 = self.classifier4(x)
        x5 = self.classifier5(x)
        x6 = self.classifier6(x)
        x7 = self.classifier7(x)
        x8 = self.classifier8(x)
        x9 = self.classifier9(x)
        x0 = x0.view(x0.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x5 = x5.view(x5.size(0), -1)
        x6 = x6.view(x6.size(0), -1)
        x7 = x7.view(x7.size(0), -1)
        x8 = x8.view(x8.size(0), -1)
        x9 = x9.view(x9.size(0), -1)
        
        return torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9], dim=1)
        """
        #x = x.view(x.size(0), -1)
        return torch.sigmoid(x)


if __name__ == "__main__":
    net = Resnet34()
    print(net)
