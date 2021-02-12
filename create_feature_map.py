#######################################
### Autor: teriterance(Gabin FODOP)####
#######################################

import torch
import cv2
from siamesenet import SiameseNet
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import sys

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