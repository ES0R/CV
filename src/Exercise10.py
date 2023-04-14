# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:03:48 2023

@author: Emil
"""

#%%
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



#%% 10.1

import cv2

# Read the images
im1 = cv2.imread("data/im1.jpg")
im2 = cv2.imread("data/im2.jpg")

im1 = im1[:,:,::-1]
im2 = im2[:,:,::-1]

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# Create a brute-force matcher object with cross-checking
bf = cv2.BFMatcher(crossCheck=True)

# Match descriptors of keypoints in both images
matches = bf.match(des1, des2)

# Filter out good matches based on a distance threshold
good = []
for m in matches:
    if m.distance < 0.0107* min(len(kp1), len(kp2)):
        good.append(m)

q1 = []
q2 = []
for match in matches:
    q1.append(kp1[match.queryIdx].pt)
    q2.append(kp2[match.trainIdx].pt)


# Draw the matched keypoints on the images
plt.figure()
plt.imshow(cv2.drawMatches(im1, kp1, im2, kp2, good, None))



#%% 10.2


best_homography = None
best_num_inliers = 0

N=200
threshold_distance = 0.3

im1 = cv2.imread("data/im1.jpg")
im2 = cv2.imread("data/im2.jpg")

im1 = im1[:,:,::-1]
im2 = im2[:,:,::-1]

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# Create a brute-force matcher object with cross-checking
bf = cv2.BFMatcher(crossCheck=True)

# Match descriptors of keypoints in both images
matches = bf.match(des1, des2)

# Filter out good matches based on a distance threshold
good = []
for m in matches:
    if m.distance < 0.0107 * min(len(kp1), len(kp2)):
        good.append(m)


for n in range(N):
    random_matches = np.random.choice(matches,4,replace=False) 
    # Compute homography using four random matches
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in random_matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in random_matches ]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold_distance)

    # Compute the number of inliers using the homography
    num_inliers = np.sum(mask != 0)

    # Update the best homography if we found more inliers
    if num_inliers > best_num_inliers:
        best_homography = homography
        best_num_inliers = num_inliers
        
p1 = []
p2 = []

for match in matches:
    p1.append(kp1[match.queryIdx].pt)
    p2.append(kp2[match.trainIdx].pt)

plt.figure()
plt.imshow(cv2.drawMatches(im1, kp1, im2, kp2, np.array(good), None))




