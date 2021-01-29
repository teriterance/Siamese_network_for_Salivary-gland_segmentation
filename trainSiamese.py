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


dataset_texture_test = TextureTestDataset(status = "test")
dataset_parotide_test = ParotideData(status = "test")

train_dataloader_texture_test = DataLoader(dataset_texture,
                        shuffle=Config.shuffle,
                        num_workers=Config.num_workers,
                        batch_size=Config.test_batch_size)

train_dataloader_parotide_test = DataLoader(dataset_parotide,
                        shuffle=Config.shuffle,
                        num_workers=Config.num_workers,
                        batch_size=Config.test_batch_size)

train_dataloader_test = train_dataloader_texture_test


net = None
cuda = torch.cuda.is_available()
if cuda:
    net = SiameseNet().cuda()
else :
    net = SiameseNet()
criterion = ContrastiveLoss(margin=Config.contrastiveloss_margin, alpha=Config.contrastiveloss_alpha, beta=Config.contrastiveloss_beta)
optimizer = optim.Adam(net.parameters(),lr = Config.lr)

counter = []
train_loss_history = [] 
val_loss_history = [] 

for epoch in range(0,Config.train_number_epochs):
    train_loss = 0
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

        train_loss = train_loss + loss_contrastive.item()

    train_loss = train_loss/Config.train_batch_size
    train_loss_history.append(train_loss)
    print("Epoch number {}\n Current loss {}\n".format(epoch, train_loss))
    
    val_loss = 0
    for data1, data2 in zip(train_dataloader_test, train_dataloader_test):
        img0, label0 = data1['image'].float().unsqueeze(1), data1['cat'] 
        img1, label1 = data2['image'].float().unsqueeze(1), data2['cat']
        label = None
        if torch.all(label0.eq(label1)):
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)

        if cuda:
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

        output1,output2 = net(img0,img1)
        loss_contrastive = criterion(output1,output2,label)
        val_loss = val_loss + loss_contrastive.item()

    val_loss = val_loss/Config.test_batch_size
    val_loss_history.append(val_loss)
    print("Epoch number {}\n Current vall loss {}\n".format(epoch, val_loss))
    
train_loss_history = np.array(train_loss_history)
val_loss_history = np.array(val_loss_history)

plt.plot(train_loss_history,label='train loss')
plt.plot(val_loss_history,label='test loss')
plt.title("Loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(1 - train_loss_history,label='train accuracy')
plt.plot(1 - val_loss_history,label='test accuracy')
plt.title("accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()