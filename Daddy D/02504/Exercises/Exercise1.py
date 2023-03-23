# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:38:56 2023

@author: danie
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import box3d, projectpoints

#%% EXERCISE 1

im = cv2.imread("photo.jpg")

im = (im[:,:,::-1])

im = im.astype(float)/255

plt.imshow(im)

#%%

box = box3d()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(box[0],box[1],box[2])

#%%

K = np.array([[1,0,0],[0,1,0],[0,0,1]])

R = np.array([[1,0,0],[0,1,0],[0,0,1]])

t = np.transpose([np.array([0,0,4])])

project = projectpoints(K, R, t, box)

plt.scatter(project[0],project[1])

#%%

theta = 30*np.pi/180


R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

projectnew = projectpoints(K, R, t, box)

plt.scatter(projectnew[0],projectnew[1])