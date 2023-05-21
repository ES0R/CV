# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:18:42 2023

@author: danie
"""

import cv2
import numpy as np
from functions import *
#%% 3

K = np.array([[350,350*0,700],[0,1*350,390],[0,0,1]])

R = cv2.Rodrigues(np.array([0.1, -0.2, -0.1]))[0]

t = np.array([[0.03], [0.06], [-0.02]])

Q = np.array([0.35, 0.17, 1.01]).reshape(3,1)

p = projectpoints(K, R, t, Q)

#%%

def smoothedHessian2(im, sigma, epsilon):
    g_ep, g_epd = gaussian1DKernel(sigma, size=4)
    g_ep2 = np.outer(g_ep,g_ep)
    I, Ix, Iy = gaussianSmoothing(im, sigma)
    c11 = scipy.ndimage.convolve(Ix**2, g_ep2)
    c12 = scipy.ndimage.convolve(Ix*Iy, g_ep2)
    c21 = scipy.ndimage.convolve(Ix*Iy, g_ep2)
    c22 = scipy.ndimage.convolve(Iy**2, g_ep2)
    C = [[c11, c12], [c21, c22]]
    return C

def harrisMeasure2(im, sigma, epsilon, k):
    C = smoothedHessian(im, sigma, epsilon)
    a = C[0][0]
    c = C[0][1]
    b = C[1][1]
    r = a*b-c**2-k*(a+b)**2
    return r
  
def threshold_neighbors2(image):
    thresholded = image.copy()
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if (image[i+1, j] >= image[i,j] or
                image[i-1, j] >= image[i,j] or
                image[i, j+1] >= image[i,j] or
                image[i, j-1] >= image[i,j]):
                thresholded[i, j] = 0
    return thresholded
  
def cornerDetector2(im, sigma, epsilon, k, tau):
    r = harrisMeasure(im, sigma, epsilon, k)
    r[r < tau] = 0
    thr = threshold_neighbors(r)
    #thr[thr > tau] = 1
    c = np.where(thr!=0)
    tempy = c[0]
    tempx = c[1]
    c = [tempx, tempy]
    return c

#%% 4

C = np.load("C:/Users/danie/OneDrive/Desktop/Master - 2. semester/02504/Exam/SampleExamQuestions/materials/harris.npy", allow_pickle=True).item()

a = C["g*(I_x^2)"]
c = C["g*(I_x I_y)"]
b = C["g*(I_y^2)"]
k = 0.06

r = a*b-c**2-k*(a+b)**2
tau = 0.2*np.max(r)
r[r < tau] = 0
thr = threshold_neighbors(r)
corner = np.where(thr!=0)
tempy = corner[0]
tempx = corner[1]
corner = [tempx, tempy]

