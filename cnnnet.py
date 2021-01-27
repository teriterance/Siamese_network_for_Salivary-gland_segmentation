import torch
import torch.nn as nn
import torch.nn.functional as F

#test 
import numpy as np
class CNNNet(nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(1, 10, 5), #1*32*32 -> 10*28*28
            nn.MaxPool2d(2), #1*28*28 -> 10*14*14
            nn.ReLU(),
            nn.Conv2d(10, 20, 5), #10*14*14 -> 20*10*10
            nn.MaxPool2d(2), #20*10*10 -> 20*5*5 = 500
            nn.ReLU(),
        )
        self.out_part = nn.Sequential(
            nn.Linear(500, 100), 
            nn.ReLU(), 
            nn.Linear(100, 20), 
            nn.ReLU(), 
            nn.Linear(20, 10)
        )
        self.sofmax = nn.Softmax(dim=1)

    def forward(self, x):
        #forward
        x = self.conv_part(x)
        x = x.view(x.size()[0], -1)
        x = self.out_part(x)
        x = self.sofmax(x)
        return x


# for test
if __name__ == '__main__':
    net = CNNNet()
    print(net)
    
    img = torch.from_numpy(np.random.rand(1,32,32)).float().unsqueeze(0)
    out = net(img)
    print(out)