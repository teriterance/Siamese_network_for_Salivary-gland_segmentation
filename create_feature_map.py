import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from siamesenet import SiameseNet
from contrastiveLoss import ContrastiveLoss
import time
from torchvision import transforms
import torch.nn as nn


def load_model(path='../model'):
    model = SiameseNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def one_feature(image_ref_pad, model):
    print(image_ref_pad.shape)
    [size_x, size_y] = image_ref_pad.shape
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
            out1, out2 = model.forward_one(imagette_1)
            img_out_1[i,j] = out1.detach().numpy() 
            img_out_2[i,j] = out2.detach().numpy()

    return img_out_1, img_out_2

"""
def gen_feature_vector(img ,model, im_size = 32):
    list_images = []
    for r in range(0,img.shape[0],im_size):
        for c in range(0,img.shape[1],im_size):
            list_images.append(img[r:r+im_size, c:c+im_size,:])
    
    for i in range 
"""

if __name__ == '__main__':
    model = load_model()
    print(model)

    img  = cv2.imread("/home/gabin/Cours/Article interessant/Recherche ali mansour/model/data_base/P01_kalimari/P01_D001_PAG_N_Echo.bmp", 0)

    img = img[65:353, 203:747].copy()
    start_time = time.time()
    img_out = one_feature(img, model)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.imshow(img_out)
    plt.show()
