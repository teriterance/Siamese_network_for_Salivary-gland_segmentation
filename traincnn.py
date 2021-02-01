from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
###
from contrastiveLoss import ContrastiveLoss
from glandeDataset import ParotideData
from cnnnet import CNNNet
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
    net = CNNNet().cuda()
else :
    net = CNNNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = Config.lr)

counter = []
train_loss_history = [] 
val_loss_history = [] 

min_loss = np.inf

for epoch in range(0,Config.train_number_epochs):
    train_loss = 0
    for data in train_dataloader:
        img, label = data['image'].float().unsqueeze(1), torch.tensor(data['cat'])
        print(label)
        if cuda:
            img , label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net(img)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()

    train_loss = train_loss/Config.train_batch_size
    train_loss_history.append(train_loss)
    print("Epoch number {}\n Current loss {}\n".format(epoch, train_loss))
    
    val_loss = 0
    for data in train_dataloader_test:
        img, label = data['image'].float().unsqueeze(1), torch.tensor(data['cat'])
        if cuda:
            img, label = img.cuda(), label.cuda()

        output = net(img)
        loss = criterion(output,label)
        val_loss = val_loss + loss.item()

    val_loss = val_loss/Config.test_batch_size
    val_loss_history.append(val_loss)
    print("Epoch number {}\n Current vall loss {}\n".format(epoch, val_loss))

    if min_loss > val_loss:
        min_loss = val_loss
        torch.save(net.state_dict(), '../model')
    
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