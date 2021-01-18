import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNNet(nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, 5),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),    # 12 8@42*42
            nn.MaxPool2d(2),   # 128@21*21
        )
        self.out = nn.Sequential(
            nn.Linear(9216, 100), 
            nn.ReLU(inplace=True), 
            nn.Linear(100, 20), 
            nn.ReLU(inplace=True), 
            nn.Linear(20, 10)
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.out(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))