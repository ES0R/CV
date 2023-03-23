# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:58:50 2023

@author: danie
"""

#%% IMPORTS

from functions import gaussian1DKernel, gaussianSmoothing, smoothedHessian
from functions import harrisMeasure, cornerDetector
import cv2
import matplotlib.pyplot as plt
#%% 6.1

g, gd = gaussian1DKernel(3, size=1)

#%% 6.2

im = cv2.imread("week06_data/week06_data/TestIm1.png")

im = (im[:,:,::-1])

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im = im.astype(float)/255

#%%

I, Ix, Iy = gaussianSmoothing(im, 5)

plt.imshow(Ix)

#%% 6.3

#g_ep, g_epd = gaussian1DKernel(5, size=4)
#g_ep2 = np.outer(g_ep,g_ep)
#c11 = scipy.ndimage.convolve(Ix**2, g_ep2)

C = smoothedHessian(im, 5, 2)

plt.imshow(C[1][1])

#%% 6.4 

r = harrisMeasure(im, 5, 2, 0.06)

plt.imshow(r)

#%% 6.5
'''
def threshold_neighbors(image):
    thresholded = image.copy()
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if (image[i+1, j] >= image[i,j] or
                image[i-1, j] >= image[i,j] or
                image[i, j+1] >= image[i,j] or
                image[i, j-1] >= image[i,j]):
                thresholded[i, j] = 0
    return thresholded

r = harrisMeasure(im, 5, 2, 0.06)

tau = 1*10**(-10)

r[r < tau] = 0

thr = threshold_neighbors(r)

thr[thr > tau] = 1

lm = np.where(thr==1)


ll = np.sum(thr)

plt.imshow(thr)

'''
#%%

#plt.scatter(lm[0], lm[1])

#%% 6.5

c = cornerDetector(im, 5, 2, 0.06, 1*10**(-10))

plt.figure()
plt.imshow(im)
plt.scatter(c[0], c[1])

#%% 6.6

img = cv2.imread("week06_data/week06_data/TestIm1.png")
# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold
  
# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)
  
plt.figure()
plt.imshow(edge)
