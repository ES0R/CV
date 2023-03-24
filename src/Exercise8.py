from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import itertools as it
from Help_func import *
import scipy
import skimage
#%% 8.1

im = cv2.imread("data/sunflowers.jpg")
im = im[:, :, ::-1]
im = im.astype(float).mean(2) / 255

plt.figure()
plt.imshow(im, cmap='gray')

#g_list = scaleSpaced(im, 3, 5)

#%% 8.2

dog = differenceOfGaussians(im, 2, 3)

#%% 8.3

def detectBlobs(im, sigma, n, tau):
    im_scales = scaleSpaced(im, sigma, n)
    

    DoG = differenceOfGaussians(im, sigma, n)
        
    maxDoG = []
    for i in range(n-2):
        dilated = cv2.dilate(np.abs(DoG[i]), np.ones((3,3)))
        
        maxima = (DoG[i] > tau) & (DoG[i] == dilated)
        
        maxDoG.append(maxima.astype(np.float32) * (i+1))
    
    maxima = np.dstack(maxDoG).max(axis=2)
    
    blobs = []
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if maxima[y,x] > 0:
                scale = maxima[y,x]
                radius = int(scale)
                blobs.append((x,y,radius))
    
    return blobs


sigma = 2
n = 7
tau = 0.1

blobs = detectBlobs(im, sigma, n, tau)

blobs = np.array(blobs)

radi = blobs[:, 2]
blobs = blobs[:, :2]

# Create a copy of the image to draw circles on
im_circles = np.copy(im)

for blob, radius in zip(blobs, radi):
    cv2.circle(im_circles, tuple(blob), int(2*sigma**(radius)), (1, 0, 0), thickness=2)

plt.figure()
plt.imshow(im_circles, cmap = "gray")
plt.show()



