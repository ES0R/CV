# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:14:41 2023

@author: danie
"""

#%% IMPORTS

import numpy as np
from functions import projectpoints, triangulate, triangulate_nonlin
import cv2
import scipy

#%% EXERCISE 5.1

R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

t1 = np.transpose([np.array([0,0,1])])
t2 = np.transpose([np.array([0,0,20])])

K1 = np.array([[700, 0, 600], [0, 700, 400], [0, 0, 1]])
K2 = np.array([[700, 0, 600], [0, 700, 400], [0, 0, 1]])

Q = np.transpose([np.array([1,1,0])])

Rt1 = np.concatenate((R1,t1), axis=1)
Rt2 = np.concatenate((R2,t2), axis=1)


P1 = K1@Rt1
P2 = K2@Rt2

q1 = projectpoints(K1, R1, t1, Q)
q2 = projectpoints(K2, R2, t2, Q)

#%% EXERCISE 5.2

q1tilde = np.add(q1, np.transpose(np.array([[1,-1]])))

q2tilde = np.add(q2, np.transpose(np.array([[1,-1]])))

q_list = [q1tilde, q2tilde]

P_list = [P1, P2]

Qt = triangulate(q_list, P_list).reshape(3,1)

q1t = projectpoints(K1, R1, t1, Qt)
q2t = projectpoints(K2, R2, t2, Qt)

e1 = np.linalg.norm(Q-Qt)

#%% EXERCISE 5.3
import scipy
print(scipy.__version__)

#%% EXERCISE 5.4


Qhat = triangulate_nonlin(q_list, P_list) 

Qhat = Qhat.reshape(3,1)

e2 = np.linalg.norm(Q-Qhat)

#%% EXERCISE 5.5
#%% EXERCISE 5.6

im1 = cv2.imread("billede1.jpg")
im2 = cv2.imread("billede2.jpg")
im3 = cv2.imread("billede3.jpg")
im4 = cv2.imread("billede4.jpg")


im1 = (im1[:,:,::-1])
im2 = (im2[:,:,::-1])
im3 = (im3[:,:,::-1])
im4 = (im4[:,:,::-1])

#%%

im_small_1 = cv2.resize(im1, None, fx=0.25, fy=0.25)
ret1, corners1 = cv2.findChessboardCorners(im_small_1, (11,8))

im_small_2 = cv2.resize(im2, None, fx=0.25, fy=0.25)
ret2, corners2 = cv2.findChessboardCorners(im_small_2, (11,8))

im_small_3 = cv2.resize(im3, None, fx=0.25, fy=0.25)
ret3, corners3 = cv2.findChessboardCorners(im_small_3, (11,8))

im_small_4 = cv2.resize(im4, None, fx=0.25, fy=0.25)
ret4, corners4 = cv2.findChessboardCorners(im_small_4, (11,8))













 