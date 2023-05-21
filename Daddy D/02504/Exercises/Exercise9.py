# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:50:44 2023

@author: danie
"""

import numpy as np
from functions import PiInv, Fest_8point2, CrossOp, projectpoints, estfundamental
#%% 9.1

res_data = np.load('Fest_test.npy', allow_pickle=True).item()

Ftrue = res_data["Ftrue"]
q1 = res_data["q1"]
q2 = res_data["q2"]

   
F = Fest_8point2(q1, q2)*-0.0002169339943618159
#%% 9.2



#%% 9.3

data = np.load('TwoImageData.npy', allow_pickle=True).item()

t2 = data["t2"]

R2 = data["R2"]

t1 = data["t1"]

R1 = data["R1"]

K = data["K"]

E = CrossOp(t2[:, 0])@R2

Ftrue = np.transpose(np.linalg.inv(K))@E@np.linalg.inv(K)

Q = np.random.rand(3,30)

q1 = PiInv(projectpoints(K, R1, t1, Q))

q2 = PiInv(projectpoints(K, R2, t2, Q))

original_array = np.random.rand(3, 30)

# Generate noise with a normal distribution
mu = 0  # Mean of the noise
sigma = 10  # Standard deviation of the noise
noise = np.random.normal(mu, sigma, q2.shape)

q2_noise = q2 + noise

F = Fest_8point2(q1, q2_noise)

Fscaled = F/(F[2,2]/Ftrue[2,2])

#%%

Fransac = estfundamental(q1, q2_noise)
Fransacscaled = Fransac/(Fransac[2,2]/Ftrue[2,2])

#%%

q1 = np.random.rand(3, 30)
q2 = np.random.rand(3, 30)

Ft = Fest_8point2(q1, q2)

mu = 0  # Mean of the noise
sigma = 10  # Standard deviation of the noise
noise = np.random.normal(mu, sigma, q2.shape)

q2_noise = q2 + noise

F8 = Fest_8point2(q1, q2_noise)
F8 = F8/(F8[2,2]/Ft[2,2])

Fr = estfundamental(q1, q2_noise)
Frs = Fr/(Fr[2,2]/Ft[2,2])