from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import cv
from PIL import Image, ImageDraw
import itertools as it
from Help_func import *
import scipy
import skimage
#%% 8.1

im = cv2.imread("data/sunflowers.jpg")
    

#g_list = scaleSpaced(im, 3, 5)

#%% 8.2

dog = differenceOfGaussians(im, 2, 3)

#%% 8.3

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
    cv2.circle(im_circles, tuple(blob), radius, (1, 0, 0), thickness=2)

plt.figure()
plt.imshow(im_circles, cmap = "gray")
plt.show()


#%% 8.4
r_im = transformIm(im,45,1)
cv2.imshow('Original image', im)
cv2.imshow('Rotated image', r_im)
cv2.waitKey(0)
def transformIm(im,theta,s):
    height, width = im.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=s)
 
    # rotate the image using cv2.warpAffine
    r_im = cv2.warpAffine(src=im, M=rotate_matrix, dsize=(width, height))
    
    return r_im

#%% 8.5

new_data = np.load('data/TwoImageData.npy', allow_pickle=True).item()

img1 = new_data["im1"]

img2 = transformIm(img1,30,1.5)


# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.2*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()



