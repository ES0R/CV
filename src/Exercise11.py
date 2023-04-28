# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:58:18 2023

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
import cv2

#%% 11.1

K = np.loadtxt('data\\Glyp\\K.txt')


im0 = cv2.imread("data/Glyp/sequence/000001.png")
im1 = cv2.imread("data/Glyp/sequence/000002.png")
im2 = cv2.imread("data/Glyp/sequence/000003.png")

im0 = im0[:,:,::-1]
im1 = im1[:,:,::-1]
im2 = im2[:,:,::-1]

plt.figure()
plt.imshow(im0)
plt.figure()
plt.imshow(im1)
plt.figure()
plt.imshow(im2)
plt.show()

sift = cv2.SIFT_create(nfeatures = 2000)

kp0, des0 = sift.detectAndCompute(im0, None)
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

temp_kp1 = kp1


#kp0 = np.array([k.pt for k in kp0])
#kp1 = np.array([k.pt for k in kp1])
#kp2 = np.array([k.pt for k in kp2])


bf_0 = cv2.BFMatcher(crossCheck=True)
bf_1 = cv2.BFMatcher(crossCheck=True)


# Match descriptors of keypoints in both images
matches01 = bf_0.match(des0, des1)
matches12 = bf_1.match(des1, des2)

matches01 = sorted(matches01, key = lambda x:x.distance)
matches12 = sorted(matches12, key = lambda x:x.distance)


matches01 = np.array([(m.queryIdx, m.trainIdx) for m in matches01])

matches12 = np.array([(m.queryIdx, m.trainIdx) for m in matches12])




#%% 11.2

#matches01
kp0_s = [kp0[i-1] for i in matches01[:,0]]
kp1_s = [kp1[i-1] for i in matches01[:,0]]

kp0_s = np.array([k.pt for k in kp0_s])
kp1_s = np.array([k.pt for k in kp1_s])

Essential_Mat_01, mask = cv2.findEssentialMat(kp0_s, kp1_s, K)

points, R, t, mask = cv2.recoverPose(Essential_Mat_01, kp0_s, kp1_s)

matches01_inliers = matches01[mask.ravel() == 255]


#matches02
kp1_s = [kp1[i-1] for i in matches12[:,0]]
kp2_s = [kp2[i-1] for i in matches12[:,0]]

kp1_s = np.array([k.pt for k in kp1_s])
kp2_s = np.array([k.pt for k in kp2_s])

Essential_Mat_12, mask = cv2.findEssentialMat(kp1_s, kp2_s, K)

points, R, t, mask = cv2.recoverPose(Essential_Mat_12, kp1_s, kp2_s)

matches12_inliers = matches01[mask.ravel() == 255]


#%% 11.3

_, idx01, idx12 = np.intersect1d(matches01_inliers[:,1], matches12_inliers[:,0], return_indices=True)

matches01_subset = matches01_inliers[idx01]
matches12_subset = matches12_inliers[idx12]

points0 = np.array([kp0[m[0]].pt for m in matches01_subset])
points1 = np.array([kp1[m[1]].pt for m in matches01_subset])
points2 = np.array([kp2[m[1]].pt for m in matches12_subset])

fig, (ax0, ax1) = plt.subplots(ncols=2)

# Display the two images on the subplots
ax0.imshow(im0)
ax1.imshow(im1)

# Plot the two points on top of the respective images

i = 

ax0.plot(points1[i][0], points1[i][1], 'ro')  # 'ro' means red circle
ax1.plot(points2[i][0], points2[i][1], 'bo')  # 'bo' means blue circle

# Show the figure
plt.show()








