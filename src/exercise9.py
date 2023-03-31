# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:50:44 2023

@author: hujo8
"""

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
#%% 9.1

res_data = np.load('Fest_test.npy', allow_pickle=True).item()

F_res = res_data["Ftrue"]
q1 = res_data["q1"]
q2 = res_data["q2"]

   
F_est = Fest_8point(q1, q2)

#%% 9.2
#%% 9.3

new_data = np.load('TwoImageData_new.npy', allow_pickle=True).item()


q1 = res_data["q1"]

q1 = unhomo(q1)

N = q1.shape[1]
index = np.linspace(0,N-1,N).astype(int)

#q2 = q1[:,np.random.choice(index, 8, replace=False)]

q2 = q1[:,[1,0,2,3,4,5,6,7]]

F_est = Fest_8point(q1, q2)

samp = []
for i in range(N):
    
    samp.append(cv2.sampsonDistance(q1[:,i],q2[:,i], F_est))

samp_index = [i for i, x in enumerate(samp) if x <= 3.84*(3**2)]

inliers1 = q1[:,samp_index]
inliers2 = q2[:,samp_index]


F_est = Fest_8point(inliers1, inliers2)