# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:36:13 2023

@author: Emil
"""

#%%
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
import cv2
import os

#%%

c = np.load('data/casper/casper/calib.npy', allow_pickle=True).item()



im0 = cv2.imread("data/casper/casper/sequence/frames0_0.png")
size = (im0.shape[1], im0.shape[0])
stereo = cv2.stereoRectify(c['K0'], c['d0'], c['K1'],
                           c['d1'], size, c['R'], c['t'], flags=0)
R0, R1, P0, P1 = stereo[:4]
maps0 = cv2.initUndistortRectifyMap(c['K0'], c['d0'], R0, P0, size, cv2.CV_32FC2)
maps1 = cv2.initUndistortRectifyMap(c['K1'], c['d1'], R1, P1, size, cv2.CV_32FC2)


folder_path = "data/casper/casper/sequence"

im0 = []
im1 = []

for filename in os.listdir(folder_path):
    if filename.startswith("frames0"):
    
    elif filename.startswith("frames1"):

#%%

print([2:17])
