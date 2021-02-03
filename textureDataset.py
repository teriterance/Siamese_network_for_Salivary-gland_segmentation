import torch
import os 
from skimage import io
import random
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset


class TextureTestDataset(Dataset):
    def __init__(self, root_dir = "../data_texture_small/", status = "train", transforms = None, test_size = 0.3):
        """transformation a appliquer au dataset, et dossier de ce dernier 
        :root_dir =  dossier du dataset 
        :status = 
        :transform = transformation a apliquer sur chaque element  
        """
        self.root_dir = root_dir
        self.transform = transforms
        #remplissage d'une liste d'element du datasset
        self.files = []
        
        for ( _ , _ , filenames) in os.walk(self.root_dir):
            self.files.extend(filenames)
        
        if status == 'test':
            a = self.__len__()
            a = int(a*test_size)
            self.files = self.files[a:].copy()

        random.shuffle(self.files)
    
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
        image = io.imread(self.root_dir+img_name, as_gray=True)
        sample = {'image': image, 'cat': int(self.files[idx].split("_")[0])}

        if self.transform:
            sample = {'image': self.transform(image), 'cat': int(self.files[idx].split("_")[0])}

        return sample


if __name__ == "__main__":
    gabinDataset  =  TextureTestDataset()
    image = gabinDataset[0]
    print(image["image"].shape, image["cat"])
    
    plt.imshow(image['image'], cmap='gray')
    plt.title(image['cat'])
    plt.show()
