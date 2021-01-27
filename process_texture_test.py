from os import read, walk, listdir, mkdir
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
import numpy as np
import cv2

def explore_img():
    img = cv2.imread("../test_texture/D1.tif")
    print(img.shape)
    print("\n\n")
    print(img)

def generate_dataset(BaseDatasetFolderPath = "../Normalized_Brodatz/", customDatasetFolderPath = "../data_texture/", imgsize = 640):
    ##### Load Image #####
    imax = 4
    imtt = 0
    counter = 0
    img_boat = [None, None, None, None]
    for texture in listdir(BaseDatasetFolderPath):
        if counter < imax:
            img_boat[counter] = cv2.imread(BaseDatasetFolderPath + texture)
            counter = counter+1
        else:
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        for l in range(4):
                            if i!=j and k!=l and i!=l and j!=k :
                                img = np.concatenate((img_boat[i][0:320,0:320].copy(), img_boat[j][0:320,0:320].copy()), axis=0)
                                img2 = np.concatenate((img_boat[l][0:320,320:].copy(), img_boat[k][0:320,320:].copy()), axis=0)
                                img_comc = np.concatenate((img.copy(), img2.copy()), axis=1)
                                print(img_comc.shape)
                                if exists(customDatasetFolderPath+str(imgsize)):
                                    pass
                                else:
                                    mkdir(customDatasetFolderPath+ str(imgsize))
                                    imtt = imtt+1
                                cv2.imwrite(join(customDatasetFolderPath, "img"+str(imtt)+".png"), img_comc)

if __name__ == "__main__":
    generate_dataset()