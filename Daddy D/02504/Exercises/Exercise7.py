# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:16:15 2023

@author: danie
"""

#%% IMPORTS

import cv2
import matplotlib.pyplot as plt
import skimage
from skimage.transform import hough_line_peaks
from functions import DrawLine
from functions import line, in_out, consensus, draw_two_points
import numpy as np
#%% 7.1

img = cv2.imread("week06_data/week06_data/Box3.bmp")

img = (img[:,:,::-1])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = img.astype(float)/255

# Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold
  
# Applying the Canny Edge filter
edges = cv2.Canny(img, t_lower, t_upper)
  
plt.figure()
plt.imshow(edges)

#%% 7.2

hspace, angles, dists = skimage.transform.hough_line(edges)

#%% 7.3

extent = [angles[0], angles[-1], dists[-1], dists[0]]
plt.imshow(hspace, extent=extent, aspect='auto')

#%% 7.4

extH, extAngles, extDists = hough_line_peaks(hspace, angles, dists)

plt.figure()
plt.imshow(hspace, extent=extent, aspect='auto')
plt.scatter(extAngles,extDists)

#%% 7.5

l_list = []

for i in range(len(extAngles)):
    l = np.array([np.cos(extAngles[i]), np.sin(extAngles[i]), -extDists[i]])
    l_list.append(l)
    
#%%

plt.figure()
plt.imshow(img)
for i in range(len(l_list)):
    DrawLine(l_list[i],img.shape)

#%% RANSAC

def test_points(n_in, n_out):
    a = (np.random.rand(n_in)-.5)*10
    b = np.vstack((a, a*.5+np.random.randn(n_in)*.25))
    points = np.hstack((b, 2*np.random.randn(2, n_out)))
    return np.random.permutation(points.T).T

points = test_points(25, 4)

#%% 7.6

p1 = points[:,0].reshape(2,1)
p2 = points[:,1].reshape(2,1)

l = line(p1,p2)

x = np.linspace(-10,10)

line2 = -l[0]/l[1]*x-l[2]/l[1]
#%%
plt.figure()
plt.scatter(p1[0],p1[1], color="red")
plt.scatter(p2[0],p2[1], color="red")
plt.plot(x,line2)

#%% 7.7

inout = in_out(points, l, 0.8)

#%%

def plot_points(points, inliers_outliers):
    inliers = points[:, inliers_outliers]
    outliers = points[:, ~inliers_outliers]
    plt.scatter(inliers[0], inliers[1], c='blue')
    plt.scatter(outliers[0], outliers[1], c='red')
    plt.show()

plt.figure()
plt.plot(x,line2)
plot_points(points,inout)

#%% 7.8

con = consensus(points, l, 0.8)

#%% 7.9

rp1, rp2 = draw_two_points(points)

#%% 7.10, 11, 12 og 13

def pca_line(x): #assumes x is a (2 x n) array of points
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l@x.mean(1)))
    return l
#%%

def RANSAC(points, N=100, threshold=0.1):
    best_line = None
    best_consensus = 0
    
    
    for n in range(N):
        p1, p2 = draw_two_points(points)
        
        l = line(p1.reshape(2,1), p2.reshape(2,1))
        
        c = consensus(points, l, threshold)
        
        if c > best_consensus:
            best_consensus = c
            best_line = l
            
    inliers_outliers = in_out(points, best_line, threshold)
    
    inliers = points[:, inliers_outliers]
    outliers = points[:, ~inliers_outliers]
    plt.scatter(inliers[0,:], inliers[1,:], c='blue')
    plt.scatter(outliers[0,:], outliers[1,:], c='red')
    
    if best_line is not None:
        
        inlier_points = points[:, inliers_outliers]
        new_line = pca_line(inlier_points)
        x = np.linspace(np.min(points[0,:]), np.max(points[0,:]), 100)
        y = -(new_line[0]*x + new_line[2])/new_line[1]
        plt.plot(x, y, c='orange')
    
    plt.show()
    
    return new_line

#%%

r_line = RANSAC(points, N = 100, threshold=0.4)


#%%

def RANSAC2(points, N=100, threshold=0.1, p=0.99):
    best_line = None
    best_consensus = 0
    m = points.shape[1]
    s = 2
    
    for n in range(N):
        
        e_hat = 1 - s/m
        
        N_hat = np.log(1-p)/np.log((1-(1-e_hat)**2))

        p1, p2 = draw_two_points(points)
        
        l = line(p1.reshape(2,1), p2.reshape(2,1))
        
        c = consensus(points, l, threshold)
        
        if c > best_consensus:
            best_consensus = c
            best_line = l
            s = c
        print("n =", n)
        print("N_hat =", N_hat)
        print("e_hat =", e_hat)
        print("m =", m)
        if N_hat < n:
            break
        
    inliers_outliers = in_out(points, best_line, threshold)
    
    inliers = points[:, inliers_outliers]
    outliers = points[:, ~inliers_outliers]
    plt.scatter(inliers[0,:], inliers[1,:], c='blue')
    plt.scatter(outliers[0,:], outliers[1,:], c='red')
    
    if best_line is not None:
        
        inlier_points = points[:, inliers_outliers]
        new_line = pca_line(inlier_points)
        x = np.linspace(np.min(points[0,:]), np.max(points[0,:]), 100)
        y = -(new_line[0]*x + new_line[2])/new_line[1]
        plt.plot(x, y, c='orange')
    
    plt.show()
    
    return new_line

r_line = RANSAC2(points, N = 100, threshold=0.4, p=0.9)