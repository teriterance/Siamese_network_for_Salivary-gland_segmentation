import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

###
from cnnnet import CNNNet
from contrastiveLoss import ContrastiveLoss
from glandeDataset import ParotideData
from proces_image import img_preprocessing
from process_texture_test import explore_img, generate_dataset_big_image, generate_dataset_small_image
from siamesenet import SiameseNet
from textureDataset import TextureTestDataset
import config as Config


dataset_texture = TextureTestDataset()
dataset_parotide = ParotideData()

train_dataloader_texture = DataLoader(dataset_texture,
                        shuffle=Config.shuffle,
                        num_workers=Config.num_workers,
                        batch_size=Config.train_batch_size)

train_dataloader_parotide = DataLoader(dataset_parotide,
                        shuffle=Config.shuffle,
                        num_workers=Config.num_workers,
                        batch_size=Config.train_batch_size)

train_dataloader = train_dataloader_texture

net = None
cuda = torch.cuda.is_available()
if cuda:
    net = SiameseNet().cuda()
else :
    net = SiameseNet()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = Config.lr)

counter = []
loss_history = [] 
iteration_number= 0

for epoch in range(0,Config.train_number_epochs):
    i = 0
    for data1, data2 in zip(train_dataloader, train_dataloader):
        img0, label0 = data1['image'].float().unsqueeze(1), data1['cat'] 
        img1, label1 = data2['image'].float().unsqueeze(1), data2['cat']
        label = None
        if torch.all(label0.eq(label1)):
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)

        if cuda:
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
        i = i+1

plt.plot(counter,loss_history)
plt.show()