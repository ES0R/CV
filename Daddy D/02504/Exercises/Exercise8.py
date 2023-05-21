# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:58:07 2023

@author: danie
"""

#%% IMPORTS

import cv2
import matplotlib.pyplot as plt
from functions import scaleSpaced, differenceOfGaussians, detectBlobs, transformIm
import numpy as np
#%% BLOB Detector

im = cv2.imread("sunflowers.jpg")
im = im[:,:,::-1]
im = im.astype(float).mean(2) / 255

plt.figure()
plt.imshow(im, cmap='gray')

#%% 8.1
'''
def scaleSpaced(im, sigma, n):
    im_scales = []
    for i in range(n):
        sigma_i = sigma*2**i
        im_i, ix, iy = gaussianSmoothing(im, sigma_i)
        im_scales.append(im_i)
    return im_scales
'''
#%%

im_scales = scaleSpaced(im, 0.5, 5)

#%% 8.2
'''
def differenceOfGaussians(im, sigma, n):
    im_scales = scaleSpaced(im, sigma, n)
    
    DoG = [im_scales[i+1] - im_scales[i] for i in range(n-1)]
    
    return DoG

'''
dog = differenceOfGaussians(im, 2, 5)

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
#Ikke kør det her bullshit af Emil
r_im = transformIm(im,45,1)
cv2.imshow('Original image', im)
cv2.imshow('Rotated image', r_im)
cv2.waitKey(0)

#%% 8.5

new_data = np.load('TwoImageData.npy', allow_pickle=True).item()

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
    if m.distance < 0.7*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
