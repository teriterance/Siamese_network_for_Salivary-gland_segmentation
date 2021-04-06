#######################################
### Autor: teriterance(Gabin FODOP)####
#######################################

import torch
import os 
from skimage import io
import random
from os.path import join
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset


class ParotideData(Dataset):
    def __init__(self, root_dir = "../data", status = "train", transforms = None, test_size=0.3):
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

        for ( _ , _ , filenames) in os.walk(join(self.root_dir, "bord_glande/")):
            for file in filenames:
                self.files_bord_glande.append([join(self.root_dir, "bord_glande/", file), 0]) #0 for bord 
        
        for ( _ , _ , filenames) in os.walk(join(self.root_dir, "glande/")):
            for file in filenames:
                self.files_glande.append([join(self.root_dir, "glande/", file), 1]) # #1 for bord

        for ( _ , _ , filenames) in os.walk(join(self.root_dir, "tissu/")):
            for file in filenames:
                self.files_autre.append([join(self.root_dir, "tissu/", file),2]) #2  du tissu
        
        self.files = self.files_glande + self.files_bord_glande + self.files_autre
        random.shuffle(self.files)
        
        if status == 'test':
            a = self.__len__()
            a = int(a*test_size)
            self.files = self.files[a:].copy()
    
    def __len__(self):
        """renvoi la taille du datasset"""
        return len(self.files)

    def __getitem__(self, idx):
        """renvoi l'un element
        :idx = indice de l'element
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx][0]
        
        image = io.imread(img_name, as_gray=True)
        sample = {'image': image, 'cat': int(self.files[idx][1])}

        if self.transform:
            sample = {'image': self.transform(image), 'cat': int(self.files[idx][1])}

        return sample


if __name__ == "__main__":
    gabinDataset  =  ParotideData()
    image = gabinDataset[0]
    print(image["image"].shape, image["cat"])
    
    plt.imshow(image['image'], cmap='gray')
    plt.title(image['cat'])
    plt.show()