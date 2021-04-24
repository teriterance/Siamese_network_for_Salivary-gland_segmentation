#######################################
### Autor: teriterance(Gabin FODOP)####
#######################################

import torch
import cv2
from siamesenet import SiameseNet
from torchvision import transforms
from sklearn.decomposition import PCA
#from RAG import RAG
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import scipy.cluster.hierarchy as sch
from clustering import hieracjiclaClustering, clustering_KNN

from os import listdir, mkdir
from os.path import isfile, join, exists

def readMask(filename, BaseDatasetFolderPathAurelien = "../v3_db/"):
    nList = filename.split('_')
    name = "_".join((nList[0],nList[2]))+ "_seg.png"
    name = "".join(("../v3_db/", name))
    img = cv2.imread(name, 0)
    try:
        if img == None:
            return "", False
    except: 
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        return thresh1, True


def load_model(path='../model'):
    """This function load saved model for generate features map"""
    model = SiameseNet()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

def features(image_ref_pad, model):
    [size_x, size_y] = image_ref_pad.shape
    print("The chape of input image",size_x, size_y)
    
    img_out_1 = image_ref_pad * 0
    img_out_2 = image_ref_pad * 0
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485],
                         std=[0.229])])
    """
    transforms.Normalize(mean=[0.485],
                         std=[0.229])])"""

    for i in range(16, size_x - 16):
        for j in range(16, size_y - 16):
            
            imagette_1 = image_ref_pad[i-16:i+16, j-16:j+16].copy()
            imagette_1 = transform(imagette_1).unsqueeze(0)
            out1 = model.forward_one(imagette_1)
            out1 = out1.detach().numpy()
            img_out_1[i,j] = out1[0][0]
            img_out_2[i,j] = out1[0][1]

    return img_out_1, img_out_2

def featuresWithMask(image_ref_pad, mask,model):
    [size_x, size_y] = image_ref_pad.shape
    print("The chape of input image",size_x, size_y)
    
    img_out_1 = image_ref_pad * 0
    img_out_2 = image_ref_pad * 0
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485],
                         std=[0.229])])
    """
    transforms.Normalize(mean=[0.485],
                         std=[0.229])])"""

    for i in range(16, size_x - 16):
        for j in range(16, size_y - 16):
            if mask[i,j] >= 250:
                imagette_1 = image_ref_pad[i-16:i+16, j-16:j+16].copy()
                imagette_1 = transform(imagette_1).unsqueeze(0)
                out1 = model.forward_one(imagette_1)
                out1 = out1.detach().numpy()
                img_out_1[i,j] = out1[0][0]
                img_out_2[i,j] = out1[0][1]
            else:
                img_out_1[i,j] = 0
                img_out_2[i,j] = 0

    return img_out_1, img_out_2

def img_preprocessing(BaseDatasetFolderPath = "../data_base/", customDatasetFolderPath = "../data/", imgsize = 32):
    ##### Load Image #####
    model = load_model()
    for patient in listdir(BaseDatasetFolderPath):
        if isfile(patient) or "xlsx" in patient:
            pass
        else:
            for fileName in listdir(join(BaseDatasetFolderPath, patient)):
                if not (".bmp" in fileName or ".jpg" in fileName and "_seg" not in fileName):
                    pass
                elif (".bmp" in fileName or ".jpg" in fileName) and "_seg" not in fileName and "_Elas" not in fileName :

                    fileName2 = None
                    if "_val." in fileName:
                        fileName2 = fileName.replace("_val.","_seg.")
                    else :
                        fileName2 = fileName.split('.')[0] + "_seg." + fileName.split('.')[1]
                    no = readMask(fileName)
                    img_seg = cv2.imread(join(BaseDatasetFolderPath, patient, fileName2), 0)

                    img_seg = img_seg[65:353, 203:747].copy()

                    print(BaseDatasetFolderPath+patient+'/'+fileName)
                    img  = cv2.imread(BaseDatasetFolderPath+patient+'/'+fileName, 0)
                    ff = fileName.split('.')
                    img = img[65:353, 203:747].copy()
                    
                    print("Timer start")
                    start_time = time.time()
                    
                    if no[1] == True:
                        n1 = cv2.resize(no[0], (544,288))
                        img_out1, img_out2 = featuresWithMask(img, n1,model)
                        img_out1 = np.multiply(img_out1,n1/256)
                        img_out2 = np.multiply(img_out2,n1/256)
                    else:
                        img_out1, img_out2 = features(img, model)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    
                    f, axarr = plt.subplots(3, 1)
                    axarr[0].imshow(img)
                    axarr[0].title.set_text("original image")
                    axarr[1].imshow(img_out1)
                    axarr[1].title.set_text("first feature image")
                    axarr[2].imshow(img_out2)
                    axarr[2].title.set_text("second feature image")
                    plt.savefig("./output/ii"+patient+fileName+".png") 
                    plt.show()
                    #clustering_KNN(img, img_out1, img_out2)

def tainingTest():
    model = load_model()
    img  = cv2.imread("C:\\Users\\User\\Documents\\model\\data_base\\P01_kalimari\\P01_D001_PAG_N_Echo.bmp", 0)
    img = img[65:353, 203:747].copy()
    img = cv2.equalizeHist(img)
    
    start_time = time.time()
    print("Timer start")
    img_out1, img_out2 = features(img, model)
    print("--- %s seconds ---" % (time.time() - start_time))

    f, axarr = plt.subplots(3, 1)
    axarr[0].imshow(img)
    axarr[0].title.set_text("original image")
    axarr[1].imshow(img_out1)
    axarr[1].title.set_text("first feature image")
    axarr[2].imshow(img_out2)
    axarr[2].title.set_text("second feature image")
    plt.show()
    clustering_KNN(img, img_out1, img_out2)


if __name__ == '__main__':
    img_preprocessing()