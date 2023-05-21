# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:13:15 2023

@author: danie
"""

import matplotlib.pyplot as plt
import numpy as np
from functions import Pi, PiInv
from functions import pest, checkerboard_points, estimateHomographies
from functions import estimate_b, estimateIntrinsics, estimateExtrinsics
from functions import calibratecamera
from scipy.spatial.transform import Rotation

#%% EXERCISE 4

R = np.array([[np.sqrt(0.5),-np.sqrt(0.5),0],[np.sqrt(0.5),np.sqrt(0.5),0],[0,0,1]])

t = np.transpose([np.array([0,0,10])])

K = np.array([[1000,0,1920/2],[0,1000,1080/2],[0,0,1]])

Rt = np.concatenate((R,t), axis=1)

P = K@Rt

Q = np.array([[0,0,0,0,1,1,1,1],[0,0,1,1,0,0,1,1],[0,1,0,1,0,1,0,1]])


q = Pi(P@PiInv(Q))

#%%

Pest = pest(Q, q)

normq = np.linalg.norm(q, axis = 1)
q_norm = q / normq[:, np.newaxis]

q_est = Pi(Pest@PiInv(Q))


rmse2 = np.sqrt(np.mean(np.sum((q_norm - q_est)**2, axis=0)))


#%%

Q_ij = checkerboard_points(3, 4)

#%% EXERCISE 4-

Q_omega = checkerboard_points(10, 20)

Ra = Rotation.from_euler('xyz', [np.pi/10, 0, 0]).as_matrix()

Rb = Rotation.from_euler('xyz', [0, 0, 0]).as_matrix()

Rc = Rotation.from_euler('xyz', [-np.pi/10, 0, 0]).as_matrix()

K = np.array([[1000,0,1920/2],[0,1000,1080/2],[0,0,1]])

R = np.array([[np.sqrt(0.5),-np.sqrt(0.5),0],[np.sqrt(0.5),np.sqrt(0.5),0],[0,0,1]])

t = np.transpose([np.array([0,0,10])])

Rt = np.concatenate((R,t), axis=1)

P = K@Rt

Q_a = Ra@Q_omega
Q_b = Rb@Q_omega
Q_c = Rc@Q_omega

q_a = Pi(P@PiInv(Q_a))
q_b = Pi(P@PiInv(Q_b))
q_c = Pi(P@PiInv(Q_c))

#%%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(Q_a[0],Q_a[1],Q_a[2])
ax.scatter(Q_b[0],Q_b[1],Q_b[2])
ax.scatter(Q_c[0],Q_c[1],Q_c[2])


#%%
plt.scatter(q_a[0],q_a[1])
plt.scatter(q_b[0],q_b[1])
plt.scatter(q_c[0],q_c[1])

#%%

qs = [q_a, q_b, q_c]

Hm = estimateHomographies(Q_omega, qs)

q_at = Pi(Hm[0]@PiInv(Q_omega[:2, :]))
q_bt = Pi(Hm[1]@PiInv(Q_omega[:2, :]))
q_ct = Pi(Hm[2]@PiInv(Q_omega[:2, :]))

#%%

b = estimate_b(Hm)

Bt = np.transpose(np.linalg.inv(K))@np.linalg.inv(K)

bt = np.array([Bt[0,0], Bt[0,1], Bt[1,1], Bt[0,2], Bt[1,2], Bt[2,2]])

bt = bt / np.linalg.norm(bt)

#%%

"""

V = np.zeros((2 * len(Hm), 6))
for i in range(len(Hm)):
    H = Hm[i]
    h11, h12, h13 = H[:, 0]
    h21, h22, h23 = H[:, 1]
    h31, h32, h33 = H[:, 2]
    
    v11 = np.array([h11*h11, h11*h12+h12*h11, h12*h12, h13*h11+h11*h13,h13*h12+h12*h13,h13*h13])
    v12 = np.array([h11*h21, h11*h22+h12*h21, h12*h22, h13*h21+h11*h23,h13*h22+h12*h23,h13*h23])
    v22 = np.array([h21*h21, h21*h22+h22*h21, h22*h22, h23*h21+h21*h23,h23*h22+h22*h23,h23*h23])
    
    V[2*i, :] = v12
    V[2*i+1, :] = v11-v22

U, S, Vt = np.linalg.svd(V)
b = Vt[-1, :]

venstre = v11@bt

B = np.transpose(np.linalg.inv(K))@np.linalg.inv(K)

h1 = np.array([h11, h12, h13])

hojre = h1@B@np.transpose(h1)

"""

#%%

Kt = estimateIntrinsics(Hm)

#%%

Rs, ts = estimateExtrinsics(Kt, Hm)
    

#%%

Kt, Rs, ts = calibratecamera(qs, Q_omega)