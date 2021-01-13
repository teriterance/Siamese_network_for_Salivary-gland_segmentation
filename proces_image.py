from os import read, walk, listdir, mkdir
from os.path import isfile, join, exists
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import pandas as pd


def get_tile_images(image, width=32, height=32):
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)

    if _m != 0 or _n != 0:
        return None

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )

def img_preprocessing(BaseDatasetFolderPath = "../data_base/", customDatasetFolderPath = "../data/", imgsize = 32):
    ##### Load Image #####
    for patient in listdir(BaseDatasetFolderPath):
        if isfile(patient) or "xlsx" in patient:
            pass
        else:
            for fileName in listdir(join(BaseDatasetFolderPath, patient)):
                if not (".bmp" in fileName or ".jpg" in fileName and "_seg" not in fileName):
                    pass
                elif (".bmp" in fileName or ".jpg" in fileName) and "_seg" not in fileName and "_Elas" not in fileName :
                    img  = cv2.imread(join(BaseDatasetFolderPath, patient, fileName))
                    bord_num = 0
                    glande_num = 0
                    tissu_num = 0
                    
                    fileName2 = None
                    if "_val." in fileName:
                        fileName2 = fileName.replace("_val.","_seg.")
                    else :
                        fileName2 = fileName.split('.')[0] + "_seg." + fileName.split('.')[1]
                    
                    img_seg = cv2.imread(join(BaseDatasetFolderPath, patient, fileName2))
                    print(join(BaseDatasetFolderPath, patient, fileName2))

                    ##### Process image #####
                    # 1 extract the real image 
                    img2  = img[65:353, 203:747].copy()
                    img_seg2 = img_seg[65:353, 203:747].copy()

                    ####### Creation des mini image ####
                    tiles = get_tile_images(img2, imgsize, imgsize)
                    tiles2 = get_tile_images(img_seg2, imgsize, imgsize)

                    _nrows = int(img2.shape[0] / imgsize)
                    _ncols = int(img2.shape[1] / imgsize)

                    fig, ax = plt.subplots(nrows=_nrows, ncols=_ncols)
                    for i in range(_nrows):
                        for j in range(_ncols):
                            ax[i, j].imshow(tiles[i, j]); ax[i, j].set_axis_off()
                            ax[i, j].set_title(str([i,j]))

                    fig, ax2 = plt.subplots(nrows=_nrows, ncols=_ncols)
                    for i in range(_nrows):
                        for j in range(_ncols):
                            ax2[i, j].imshow(tiles2[i, j]); ax2[i, j].set_axis_off()
                            ax2[i, j].set_title(str([i,j]))
                    plt.show(block=False)

                    ######### Process Humain #######
                    print("Rentrer les valeurs i et j des Bord de la glande, format \"i j\" et \"N\" pour arreter")
                    imagete = "oui"
                    bord_glande_list = []
                    while imagete != "N":
                        imagete = input()
                        if imagete != "N":
                            imagete = imagete.split()
                            bord_glande_list.append([int(imagete[0]), int(imagete[1])])

                    print("Rentrer les valeurs i et j de la glande, format \"i j\" et \"N\" pour arreter")
                    imagete = "oui"
                    glande_list = []
                    while imagete != "N":
                        imagete = input()
                        if imagete != "N":
                            imagete = imagete.split()
                            glande_list.append([int(imagete[0]), int(imagete[1])])

                    ############# enregistrement des images en question  #######
                    ## Le bord
                    if exists(customDatasetFolderPath+str(imgsize)):
                        pass
                    else:
                        mkdir(customDatasetFolderPath+ str(imgsize))
                        mkdir(join(customDatasetFolderPath, "bord_glande"))
                        mkdir(join(customDatasetFolderPath, "glande"))
                        mkdir(join(customDatasetFolderPath, "tissu"))

                    for i in range(_nrows):
                        for j in range(_ncols):
                            ## Le bord
                            if [i,j] in bord_glande_list:
                                cv2.imwrite(join(customDatasetFolderPath, "bord_glande/img"+str(bord_num)+".png"), tiles[i, j])
                                bord_num = bord_num + 1
                            ## les glandes
                            elif [i,j] in bord_glande_list:
                                cv2.imwrite(join(customDatasetFolderPath, "glande/img"+str(glande_num)+".png"), tiles[i, j])
                                glande_num = glande_num + 1
                            else:
                                cv2.imwrite(join(customDatasetFolderPath, "tissu/img"+str(tissu_num)+".png"), tiles[i, j])
                                tissu_num = tissu_num + 1
                    plt.close('all')

if __name__ == "__main__":
    img_preprocessing()