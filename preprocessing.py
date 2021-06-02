import torch
import cv2
from siamesenet import SiameseNet
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np


img = cv2.imread('data')
print(np.max(img))
print(np.min(img))

