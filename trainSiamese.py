#######################################
### Autor: teriterance(Gabin FODOP)####
#######################################

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torchvision import transforms
###
from contrastiveLoss import ContrastiveLoss
from glandeDataset import ParotideData
from siamesenet import SiameseNet
from textureDataset import TextureTestDataset
import config as Config
from create_feature_map import img_preprocessing, tainingTest


if __name__ == '__main__':
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                            std=[0.229])
    ])

    dataset_texture = TextureTestDataset(transforms=transform)
    dataset_parotide = ParotideData(transforms=transform)
    print(len(dataset_texture))
    print(len(dataset_parotide))

    train_dataloader_texture = DataLoader(dataset_texture,
                            shuffle=Config.shuffle,
                            num_workers=Config.num_workers,
                            batch_size=Config.train_batch_size)

    train_dataloader_parotide = DataLoader(dataset_parotide,
                            shuffle=Config.shuffle,
                            num_workers=Config.num_workers,
                            batch_size=Config.train_batch_size)

    train_dataloader = train_dataloader_parotide


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

    train_dataloader_test = train_dataloader_parotide_test


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

    min_loss = np.inf

    for epoch in range(0,Config.train_number_epochs):
        train_loss = 0
        for data1, data2 in zip(train_dataloader, train_dataloader):
            img0, label0 = data1['image'].float(), data1['cat']
            img1, label1 = data2['image'].float(), data2['cat']
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
            img0, label0 = data1['image'].float(), data1['cat']
            img1, label1 = data2['image'].float(), data2['cat']
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

        if min_loss > val_loss:
            min_loss = val_loss
            torch.save(net.state_dict(), '../model')
        if (epoch+1)%5 == 0:
            tainingTest()
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