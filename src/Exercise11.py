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








sift = cv2.SIFT_create()

kp0, des0 = sift.detectAndCompute(im0, None)
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)


#kp0 = np.array([k.pt for k in kp0])
#kp1 = np.array([k.pt for k in kp1])
#kp2 = np.array([k.pt for k in kp2])


bf_0 = cv2.BFMatcher(crossCheck=True)
bf_1 = cv2.BFMatcher(crossCheck=True)


# Match descriptors of keypoints in both images
matches01 = bf_1.match(des0, des1)
matches12 = bf_0.match(des1, des2)

matches01 = sorted(matches01, key = lambda x:x.distance)
matches12 = sorted(matches12, key = lambda x:x.distance)


matches01 = np.array([(m.queryIdx, m.trainIdx) for m in matches01])

matches12 = np.array([(m.queryIdx, m.trainIdx) for m in matches12])




#%% 11.2

kp0_s = [kp0[i] for i in matches01[:,0]]
kp1_s = [kp1[i] for i in matches01[:,0]]

kp0_s = np.array([k.pt for k in kp0_s])
kp1_s = np.array([k.pt for k in kp1_s])

Essential_Mat, mask = cv2.findEssentialMat(kp0_s, kp1_s , K)

points, R, t, mask = cv2.recoverPose(Essential_Mat, kp0_s, kp1_s)




