# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:26:20 2023

@author: danie
"""

#%% imports

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
#%% 13.2

c = np.load('casper/casper/calib.npy', allow_pickle=True).item()

im0 = cv2.imread("casper/casper/sequence/frames0_0.png")
size = (im0.shape[1], im0.shape[0])
stereo = cv2.stereoRectify(c['K0'], c['d0'], c['K1'],
                           c['d1'], size, c['R'], c['t'], flags=0)
R0, R1, P0, P1 = stereo[:4]
maps0 = cv2.initUndistortRectifyMap(c['K0'], c['d0'], R0, P0, size, cv2.CV_32FC2)
maps1 = cv2.initUndistortRectifyMap(c['K1'], c['d1'], R1, P1, size, cv2.CV_32FC2)

ims0 = []
ims1 = []

folder_path = 'casper/casper/sequence'

for filename in os.listdir(folder_path):
    if filename.startswith('frames0'):
        im0 = cv2.imread(os.path.join(folder_path, filename))
        gray_im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        float_im0 = gray_im0.astype(float)/255.0
        r0 = cv2.remap(float_im0, *maps0, cv2.INTER_LINEAR)
        ims0.append(r0)
    if filename.startswith('frames1'):
        im1 = cv2.imread(os.path.join(folder_path, filename))
        gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        float_im1 = gray_im1.astype(float)/255.0
        r1 = cv2.remap(float_im1, *maps1, cv2.INTER_LINEAR)
        ims1.append(r1)
        

#%%

plt.figure()
plt.imshow(ims0[0])
plt.figure()
plt.imshow(ims1[0])

#%% 13.3

n1 = 40

n2 = n1 + 1

prim0 = ims0[2:18]
fft_primary0 = np.fft.rfft(prim0, axis=0)

theta_primary0 = np.angle(fft_primary0[1])

sec0 = ims0[18:26]
fft_sec0 = np.fft.rfft(sec0, axis=0)

theta_sec0 = np.angle(fft_sec0[1])

theta_c0 = np.mod(theta_sec0 - theta_primary0, 2*np.pi)

o_primary0 = np.round((n1*theta_c0 - theta_primary0)/(2*np.pi))

theta0 = (2*np.pi*o_primary0 - theta_primary0)/n1

#%%

plt.figure()
plt.imshow(theta0)

#%%

def unwrap(ims):
    n1 = 40

    #n2 = n1 + 1

    prim = ims[2:18]
    fft_primary = np.fft.rfft(prim, axis=0)
    
    theta_primary = np.angle(fft_primary[1])
    
    sec = ims[18:26]
    fft_sec = np.fft.rfft(sec, axis=0)
    
    theta_sec = np.angle(fft_sec[1])
    
    theta_c = np.mod(theta_sec - theta_primary, 2*np.pi)
    
    o_primary = np.round((n1*theta_c - theta_primary)/(2*np.pi))

    theta = (2*np.pi*o_primary - theta_primary)/n1
    return theta

#%%

theta0 = unwrap(ims0)

plt.figure()
plt.imshow(theta0)


