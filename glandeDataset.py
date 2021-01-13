from os.path import join
import torch
import os 
import pandas as pd 
import numpy as np
from skimage import io, transform
import random
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ParotideData(Dataset):
    def __init__(self, root_dir = "../data", status = "train", transforms = None):
        """transformation a appliquer au dataset, et dossier de ce dernier 
        :root_dir =  dossier du dataset 
        :status = 
        :transform = transformation a apliquer sur chaque element  
        """
        self.root_dir = root_dir
        self.transform = transforms
        #remplissage d'une liste d'element du datasset
        self.files_glande = []
        self.files_bord_glande = []
        self.files_autre = []

        for ( _ , _ , filenames) in os.walk(self.root_dir+"/bord_glande"):
            filenames = os.path.join(self.root_dir, "/bord_glande")
            self.files_glande.extend([filenames, 0]) #0 for bord 
        
        for ( _ , _ , filenames) in os.walk(self.file_dir+"/glande"):
            filenames = os.path.join(self.root_dir, "/glande", filenames)
            self.files_bord_glande.extend([filenames, 1]) # #1 for bord

        for ( _ , _ , filenames) in os.walk(self.file_dir+"/tissu"):
            filenames = os.path.join(self.root_dir, "/tissu", filenames)
            self.files_autre.extend([filenames,2]) #2  du tissu
        
        self.files = self.files_glande + self.files_bord_glande + self.files_autre
        self.files = random.shuffle(self.files)
    
    def __len__(self):
        """renvoi la taille du datasset"""
        return len(self.files)

    def __getitem__(self, idx):
        """renvoi l'un element
        :idx = indice de l'element
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = io.imread(img_name)
        sample = {'image': image, 'cat': int(self.files[idx][1])}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, cat = sample['image'], sample['cat']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'cat': torch.tensor(cat)}


if __name__ == "__main__":
    gabinDataset  =  ParotideData(".")
    image = gabinDataset[0]
    print(image["image"].shape, image["cat"])
    dataloader = DataLoader(gabinDataset, batch_size=4,shuffle=True, num_workers=4)