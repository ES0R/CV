# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:05:41 2023

@author: danie
"""

import numpy as np
import cv2
import scipy


def  gaussian1DKernel(sigma, size=5):
     s = np.ceil(np.max([sigma*size, size]))
     x = np.arange(-s,s+1)
     x = x.reshape(x.shape + (1,))
     g = np.exp(-x**2/(sigma*sigma))
     g = g/np.sum(g)
     dg = -x/(sigma*sigma)*g
     return g, dg
    
def gaussianSmoothing(img, sigma):
    g,dg = gaussian1DKernel(sigma, 5)
    img = scipy.ndimage.convolve(img,g.T)
    img_smo = scipy.ndimage.convolve(img,g)
    img_ydiv = scipy.ndimage.convolve(img,dg)
    img_xdiv = scipy.ndimage.convolve(img,dg.T)
    return img_smo, img_xdiv,img_ydiv   

def scaleSpaced(im, sigma, n):
    
    g_list = []
    for i in range(n):
        
        img_smo,img_xdiv,img_ydiv = gaussianSmoothing(im, sigma*2**(i))
        g_list.append(img_smo)
        #plt.figure()
        #plt.imshow(img_smo, cmap='gray')
    return g_list

def differenceOfGaussians(im, sigma, n):

    g_list = scaleSpaced(im, sigma, n)
    dog_list = [y - x for x,y in zip(g_list,g_list[1:])]
    return dog_list

def detectBlobs(im,sigma,n,tau):

    DoG = differenceOfGaussians(im, sigma, n)
    
    maxDoG = []
    for i in range(n-1):
        dilated = cv2.dilate(np.abs(DoG[i]), np.ones((3,3)))
        
        maxima = (np.abs(DoG[i]) > tau)  & (np.abs(DoG[i]) == dilated)
        
        maxDoG.append(maxima.astype(np.float32) * (i+1)) #maxima.astype(np.float32)
    
    maxima = np.dstack(maxDoG).max(axis=2)
    
    blobs = []
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if maxima[y,x] > 0:
                scale = maxima[y,x]
                radius = int(2 * sigma**(scale-1))
                blobs.append((x,y,radius))
    

    return blobs

def transformIm(im,theta,s):
    height, width = im.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=s)
 
    # rotate the image using cv2.warpAffine
    r_im = cv2.warpAffine(src=im, M=rotate_matrix, dsize=(width, height))
    
    return r_im

