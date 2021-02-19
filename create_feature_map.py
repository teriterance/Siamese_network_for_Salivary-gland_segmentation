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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

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
                         
    for i in range(16, size_x - 16):
        for j in range(16, size_y - 16):
            imagette_1 = image_ref_pad[i-16:i+16, j-16:j+16].copy()
            imagette_1 = transform(imagette_1).unsqueeze(0)
            out1 = model.forward_one(imagette_1)
            out1 = out1.detach().numpy()
            img_out_1[i,j] = out1[0][0]
            img_out_2[i,j] = out1[0][1]

    return img_out_1, img_out_2


def clustering(img, feature_1, feature_2):
    im_shape = img.shape
    print(im_shape)
    #linearisation des image
    X = np.array(list(zip(feature_1.flatten(), feature_2.flatten())))
    print(X.shape)
    #dendrogram = sch.dendrogram(sch.linkage(X[0:20000], method='ward'))
    #hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    #y_hc = hc.fit_predict(X[0:20000])
    #print(y_hc)
    #print(y_hc.shape)
    kmeans = KMeans(n_clusters=4)
    y_km = kmeans.fit_predict(X)
    print(y_km)
    print(im_shape)
    img_clust = np.reshape(y_km, im_shape)
    plt.imshow(img_clust)
    plt.show()




if __name__ == '__main__':
    model = load_model()
    print("The loaded model is: \n",model)

    for arg in sys.argv[1:]:
        print(arg)
        img  = cv2.imread(arg, 0)
        img = img[65:353, 203:747].copy()
        
        start_time = time.time()
        print("Timer start")
        img_out1, img_out2 = features(img, model)
        print("--- %s seconds ---" % (time.time() - start_time))

        f, axarr = plt.subplots(3, 1)
        axarr[0].imshow(img, cmap='gray')
        axarr[0].title.set_text("original image")
        axarr[1].imshow(img_out1, cmap='gray')
        axarr[1].title.set_text("first feature image")
        axarr[2].imshow(img_out2, cmap='gray')
        axarr[2].title.set_text("second feature image")
        plt.show()

        clustering(img, img_out1, img_out2)
