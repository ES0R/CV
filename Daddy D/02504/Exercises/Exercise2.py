# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:02:35 2023

@author: danie
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import box3d, Pi, PiInv, projectpoints, distprojectpoints
from functions import undistortImage, normalize2d, hest


#%% EXERCISE 2

K = np.array([[600,600*0,400],[0,1*600,400],[0,0,1]])

R = np.array([[1,0,0],[0,1,0],[0,0,1]])

t = np.transpose([np.array([0,0.2,1.5])])

Rt = np.concatenate((R,t), axis=1)

box = box3d()


projmatrix = K@Rt
projmatrix2 = K@Rt@PiInv(box)

project = projectpoints(K, R, t, box)

#The corner projects to [100, 220].

#No since the principal point is 800 and some exceed that limit!

#%%

K = np.array([[600,600*0,400],[0,1*600,400],[0,0,1]])

R = np.array([[1,0,0],[0,1,0],[0,0,1]])

t = np.transpose([np.array([0,0.2,1.5])])

Rt = np.concatenate((R,t), axis=1)


project = distprojectpoints(K, R, t, box,-0.2,0,0)

plt.scatter(project[0],project[1])

#Yes all captured. P1 = [120,4, 232.24]

#%%

im = cv2.imread("gopro_robot.jpg")

im = (im[:,:,::-1])

im = im.astype(float)/255

f = im.shape[1]*0.455732
deltay = im.shape[0]/2
deltax = im.shape[1]/2  
beta = 0
alpha = 1
K = np.array([[f,beta*f,deltax],[0,alpha*f,deltay],[0,0,1]])

plt.imshow(im)

#%%

imu = undistortImage(im, K, -0.2, 0, 0)

plt.imshow(imu)

#%%

H = np.array([[-2, 0, 1],[1, -2, 0],[0, 0, 3]])

p = np.array([[1, 0, 2, 2],[1, 3, 3, 4]])

q = Pi(H@PiInv(p))

#%%

T, normp = normalize2d(p)

#%%

H = hest(p,q,True)*5.448542151738266

#%%

q2 = np.random.randn(2, 100)
q2h = PiInv(q2)
H_true = np.random.randn(3,3)
q1h = H_true@q2h
q1 = Pi(q1h)

H = hest(q2,q1,True)

#%%

#H = np.array([[0.3, 0.8, 0.2],[3.0, -0.1, 0.8],[0.3, 0.1, -0.2]])

#def warpImage(im, H):
#    imWarp = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
#    return imWarp


#imWarp = warpImage(im, H)

#plt.imshow(imWarp)