# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:48:43 2023

@author: danie
"""

#%% imports

import cv2
from matplotlib import pyplot as plt
import numpy as np
from functions import Pi, hest, estHomographyRANSAC
#%% 10.1

# Load images
im1 = cv2.imread("im1.jpg")
im1 = im1[:,:,::-1]

im2 = cv2.imread("im2.jpg")
im2 = im2[:,:,::-1]

# Initiate SIFT detector
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(im1,None)
kp2, des2 = sift.detectAndCompute(im2,None)

bf = cv2.BFMatcher(crossCheck=True)

matches = bf.match(des1, des2)

# Apply ratio test
good = []
for m in matches:
    if m.distance < 0.0107* min(len(kp1), len(kp2)):
        good.append(m)

im3 = cv2.drawMatches(im1,kp1,im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(im3),plt.show()



#%% 10.2

im1 = cv2.imread("im1.jpg")
im2 = cv2.imread("im2.jpg")

im1 = im1[:,:,::-1]
im2 = im2[:,:,::-1]


best_H = None
best_num_inliers = 0

N=200

sigma = 3

threshold_distance = (5.99*sigma**2)**2

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# Create a brute-force matcher object with cross-checking
bf = cv2.BFMatcher(crossCheck=True)

# Match descriptors of keypoints in both images
matches = bf.match(des1, des2)


for n in range(N):
    random_matches = np.random.choice(matches,4,replace=False)
    # Compute homography using four random matches
    src_pts = np.array([ kp1[m.queryIdx].pt for m in random_matches ]).T
    dst_pts = np.array([ kp2[m.trainIdx].pt for m in random_matches ]).T
    H = hest(src_pts, dst_pts, True)
    
    
    good = []

    for i in matches:
        p1 = np.array([kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1], 1])
        p2 = np.array([kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1], 1])
        
        #dist = np.linalg.norm(Pi(H@p2) - Pi(p1))**2 + np.linalg.norm(Pi(np.linalg.inv(H)@p1) - Pi(p2))**2
        dist = np.linalg.norm(Pi(H@p1) - Pi(p2))**2 + np.linalg.norm(Pi(np.linalg.inv(H)@p2) - Pi(p1))**2
        if dist < threshold_distance:
            good.append(i)
            
    # Compute the number of inliers using the homography
    num_inliers = len(good)
    # Update the best homography if we found more inliers
    if num_inliers > best_num_inliers:
        best_H = H
        best_num_inliers = num_inliers



#%%

im1 = cv2.imread("im1.jpg")
im2 = cv2.imread("im2.jpg")

im1 = im1[:,:,::-1]
im2 = im2[:,:,::-1]

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

best_H = estHomographyRANSAC(kp1, des1, kp2, des2)

#%%


def warpImage(im, H, xRange, yRange):
    T = np.eye(3)
    T[:2, 2] = [-xRange[0], -yRange[0]]
    H = T@H
    outSize = (xRange[1]-xRange[0], yRange[1]-yRange[0])
    mask = np.ones(im.shape[:2], dtype=np.uint8)*255
    imWarp = cv2.warpPerspective(im, H, outSize)
    maskWarp = cv2.warpPerspective(mask, H, outSize)
    return imWarp, maskWarp

xRange = [0, im1.shape[1]]
yRange = [0, im1.shape[0]]

im = warpImage(im1, best_H, xRange, yRange)

im = im[0]

plt.figure()
plt.imshow(im2)
plt.figure()
plt.imshow(im)


#%%



