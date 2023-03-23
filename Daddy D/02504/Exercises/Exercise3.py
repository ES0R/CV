# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:07:48 2023

@author: danie
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import PiInv, projectpoints
from functions import CrossOp, triangulate
from scipy.spatial.transform import Rotation

#%% EXERCISE 3

K = np.array([[1000,0,300],[0,1000,200],[0,0,1]])

R1 = np.array([[1,0,0],[0,1,0],[0,0,1]])

t1 = np.transpose([np.array([0,0,0])])

R2 = Rotation.from_euler('xyz', [0.7, -0.5, 0.8]).as_matrix()

t2 = np.transpose([np.array([0.2,2,1])])

Q = np.transpose([np.array([1,0.5,4])])


projectq1 = projectpoints(K, R1, t1, Q)

projectq2 = projectpoints(K, R2, t2, Q)

#%%

p = np.transpose(np.array([1,2,3]))

pp = np.transpose(np.array([2,3,4]))

ppp = CrossOp(p)@pp

pppp = np.cross(p, pp)

#%%

E2 = CrossOp(t2[:, 0])@R2

F2 = np.transpose(np.linalg.inv(K))@E2@np.linalg.inv(K)

#%%

epline = F2@PiInv(projectq1)*4.000030274590537

#%%

answer = epline[0]*projectq2[0]+epline[1]*projectq2[1]+epline[2]

answer2 = np.transpose(PiInv(projectq2))@epline

#This must be true, since both the point q2 and the line l are derived from the same 3D point Q.
#This 3D point yields a single epipolar plane, and the plane yields a single line in each camera. The
#projections of the 3D point must lie on the epipolar lines.

#%%

answer36 = cv2.imread("3.6.png")

answer36 = (answer36[:,:,::-1])

answer36 = answer36.astype(float)/255

plt.imshow(answer36)

#%%

answer37 = cv2.imread("3.7.png")

answer37 = (answer37[:,:,::-1])

answer37 = answer37.astype(float)/255

plt.imshow(answer37)

#%%

data = np.load('TwoImageData.npy', allow_pickle=True).item()

t2 = data["t2"]

R2 = data["R2"]

K = data["K"]

E = CrossOp(t2[:, 0])@R2

F = np.transpose(np.linalg.inv(K))@E@np.linalg.inv(K)

#%% SEE JUPYTER NOTEBOOK FOR 3.9 AND 3.10


#%%

q1 = np.array([300,160,1]).T
q2 = np.array([300,640,1]).T
q_list = [q1, q2]


P1 = np.array([[800, 0, 300, 0], [0, 800, 400, -2400], [0, 0, 1, 0]])
P2 = np.array([[800, 0, 300, 0], [0, 800, 400, 2400], [0, 0, 1, 0]])

P_list = [P1, P2]


x = triangulate(q_list, P_list)

#%%

K = np.array([[1000,0,300],[0,1000,200],[0,0,1]])

R1 = np.array([[1,0,0],[0,1,0],[0,0,1]])

t1 = np.transpose([np.array([0,0,0])])

R2 = Rotation.from_euler('xyz', [0.7, -0.5, 0.8]).as_matrix()

t2 = np.transpose([np.array([0.2,2,1])])

Q = np.transpose([np.array([1,0.5,4])])


projectq1 = projectpoints(K, R1, t1, Q)

projectq2 = projectpoints(K, R2, t2, Q)

q_list = [projectq1, projectq2]

Rt1 = np.concatenate((R1,t1), axis=1)

Rt2 = np.concatenate((R2,t2), axis=1)

P1 = K@Rt1

P2 = K@Rt2

P_list = [P1, P2]

x = triangulate(q_list, P_list)