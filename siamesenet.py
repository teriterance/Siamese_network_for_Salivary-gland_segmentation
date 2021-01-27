import torch
import torch.nn as nn
import torch.nn.functional as F

#for test
import numpy as np


class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(1, 10, 5),  # 1*32*32 -> 10*28*28
            nn.MaxPool2d(2),  # 1*28*28 -> 10*14*14
            nn.Tanh(),
            #second conv part
            nn.Conv2d(10, 20, 5), # 10*14*14 -> 20*10*10
            nn.MaxPool2d(2),   # 20*10*10 -> 20*5*5 = 500
            nn.Tanh(),
        )
        self.out_part = nn.Sequential(
            nn.Linear(500, 100), # 500 -> 100
            nn.ReLU(), 
            nn.Linear(100, 20), # 100 -> 20
            nn.ReLU(), 
            nn.Linear(20, 2), # 20 -> 2
            nn.ReLU()
        )

    def forward_one(self, x):
        x = self.conv_part(x)
        x = x.view(x.size()[0], -1)
        x = self.out_part(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out = torch.abs(out1 - out2)
        return out

if __name__ == '__main__':
    net = SiameseNet()
    print(net)
    
    img1 = torch.from_numpy(np.random.rand(1,32,32)).float().unsqueeze(0)
    img2 = torch.from_numpy(np.random.rand(1,32,32)).float().unsqueeze(0)
    out = net(img1, img2)
    print(out)