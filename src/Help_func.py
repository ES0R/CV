# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:05:41 2023

@author: hujo8
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import itertools as it
import scipy


#%% funcs

def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

def unhomo(p):
    P = p[:-1]/p[-1]
    return P
def homo(p):
    shape = np.shape(p)

    ones = np.transpose(np.ones([shape[1],1]))

    ans = np.concatenate((p, ones), axis=0)
    return ans

def line_point_len(l,p):
    p = np.abs(np.dot(l,p))/(np.abs(p[-1])*np.sqrt(l[0]**2+l[1]**2))
    return p

def projectpoints(Q,K,R,t,k3=0,k5=0,k7=0):
    
    rt = np.concatenate((R,t),axis = 1)
    
    box = homo(Q)
    
    p = rt@box
    
    PI = K@rt
    
    x = p[0,:]
    y = p[1,:]
    r = np.sqrt(x**2+y**2)
    dist = (1+(k3*r**2+k5*r**4+k7*r**6))
    p[0,:] = x * dist
    p[1,:] = y * dist
    
    box = K@p

    Q = unhomo(box)
    
    return Q,PI

def rotationm(x,y,z):
    Rx = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    R = Rz@Ry@Rx
    return R

def crossOp(p):
    M = np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])
    return M

def DrawLine(l, shape):
    #Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2]/q[2]
        if all(q>=0) and all(q+1<=shape[1::-1]):
            return q
    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1-shape[1]], [0, 1, 1-shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    plt.plot(*np.array(P).T)
def DrawLine2(l, shape):
    #Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2]/q[2]
        if all(q>=0) and all(q+1<=shape[1::-1]):
            return q
    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1-shape[1]], [0, 1, 1-shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    if (len(P)==0):
        print("Line is completely outside image")
    plt.plot(*np.array(P).T)

    
    
def pest_point(Q,q):
    
    B = [[0,-Q[0],Q[0]*q[1],0,-Q[1],Q[1]*q[1],0,-Q[2],Q[2]*q[1],0,-1,q[1]],
         [Q[0],0,-Q[0]*q[0],Q[1],0,-Q[1]*q[0],Q[2],0,-Q[2]*q[0],1,0,-q[0]],
         [-Q[0]*q[1],Q[0]*q[0],0,-Q[1]*q[1],Q[1]*q[0],0,-Q[2]*q[1],Q[2]*q[0],0,-q[1],q[0],0]]
    B = np.array(B)

    U,D,V = np.linalg.svd(B)
    
    z = V[-1,:]
    P = V[-1,:].reshape(4,3).T
    return P,z
def pest(Q, q):
    
    n = Q.shape[1]
    
    normq = np.linalg.norm(q, axis = 1)
    
    q = q / normq[:, np.newaxis]
    
    A = np.zeros((2 * n, 12))
    for i in range(n):
        A[2 * i] = np.array([*Q[:,i], 1, 0, 0, 0, 0, -q[0,i]*Q[0,i], -q[0,i]*Q[1,i], -q[0,i]*Q[2,i], -q[0,i]])
        A[2 * i + 1] = np.array([0, 0, 0, 0, *Q[:,i], 1, -q[1,i]*Q[0,i], -q[1,i]*Q[1,i], -q[1,i]*Q[2,i], -q[1,i]])

    U, D, Vt = np.linalg.svd(A)
    P = Vt[-1,:].reshape((3, 4))

    return P

def checkerboard_points(n, m):
    Q = []
    for i in range(n-1):
        for j in range(m-1):
            Q.append(np.array([i-(n-1)/2,j-(m-1)/2,0]))

    Q = np.vstack(Q).T
    return Q
    
def normalize2d(points):
    mean = np.mean(points, axis=1).reshape((2, 1))
    std = np.std(points, axis=1).reshape((2, 1))
    T = np.array([[1/std[0, 0], 0, -mean[0, 0]/std[0, 0]],
                  [0, 1/std[1, 0], -mean[1, 0]/std[1, 0]],
                  [0, 0, 1]])
    normalized_points = np.dot(T, np.vstack((points, np.ones((1, points.shape[1])))))[:2, :]
    return T, normalized_points

def hest(q1, q2, normalize=False):
    if normalize:
        T1, q1 = normalize2d(q1)
        T2, q2 = normalize2d(q2)
    
    N = q1.shape[1]
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x, y = q1[0, i], q1[1, i]
        u, v = q2[0, i], q2[1, i]
        A[2 * i, :] = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
        A[2 * i + 1, :] = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]
    U, S, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    
    if normalize:
        H = np.dot(np.dot(np.linalg.inv(T2), H), T1)
    
    return H

def estimateHomographies(Q, qs):
    temp = []
    for i in range(len(qs)):
    
        temp.append(hest(Q[:2,:],qs[i]))
    return temp
def triangulate(q_list, P_list):
    # Construct the homogeneous linear system Ax=0
    A = np.zeros((2 * len(q_list), 4))
    for i, (q, P) in enumerate(zip(q_list, P_list)):
        A[2*i, :] = q[0] * P[2, :] - P[0, :]
        A[2*i + 1, :] = q[1] * P[2, :] - P[1, :]

    # Solve the homogeneous linear system using SVD
    U, s, Vt = np.linalg.svd(A)
    x = Vt[-1, :]
    x /= x[-1]  # Ensure that x is homogeneous

    # Return the 3D point
    return x[:-1]



def triangulate_nonlin(q_list, P_list):

    x0 = triangulate(q_list, P_list)
        
    def compute_residuals(Q):
        Q = Q.reshape(3,1)
        res_list = []
        for i in range(len(P_list)):
            res_list.append(unhomo(P_list[i]@homo(Q))-q_list[i])
        return np.array(res_list).reshape(len(P_list)*2,)
        
    res = scipy.optimize.least_squares(compute_residuals, x0)["x"]
    return res

def  gaussian1DKernel(sigma, size=5):
     s = np.ceil(np.max([sigma*size, size]))
     x = np.arange(-s,s+1)
     x = x.reshape(x.shape + (1,))
     g = np.exp(-x**2/(sigma*sigma))
     g = g/np.sum(g)
     dg = -x/(sigma*sigma)*g
     ddg = -1/(sigma*sigma)*g -x/(sigma*sigma)*dg
     return g, dg
    
def gaussianSmoothing(img, sigma):
    g,dg = gaussian1DKernel(sigma, 5)
    img = scipy.ndimage.convolve(img,g.T)
    img_smo = scipy.ndimage.convolve(img,g)
    img_ydiv = scipy.ndimage.convolve(img,dg)
    img_xdiv = scipy.ndimage.convolve(img,dg.T)
    return img_smo, img_xdiv,img_ydiv   

def smoothedHessian(img, sigma, epsilon):

    g,dg =  gaussian1DKernel(epsilon, 5)
    
    I, Ix, Iy = gaussianSmoothing(img, 5)
    
    c = []
    c.append(scipy.ndimage.convolve(scipy.ndimage.convolve(Ix**2,g.T),g))
    c.append(scipy.ndimage.convolve(scipy.ndimage.convolve(Ix*Iy,g.T),g))
    c.append(scipy.ndimage.convolve(scipy.ndimage.convolve(Ix*Iy,g.T),g))
    c.append(scipy.ndimage.convolve(scipy.ndimage.convolve(Iy**2,g.T),g))
    return c[0], c[1], c[2], c[3]

def harrisMeasure(img, sigma, epsilon, k):

    c1, c2, c3, c4 = smoothedHessian(img, sigma, epsilon)
    r = c1*c4-c3**2-k*(c1+c4)**2
    return r

def threshold_neighbors(image):
    thresholded = image.copy()
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if (image[i+1, j] >= image[i,j] or
                image[i-1, j] >= image[i,j] or
                image[i, j+1] >= image[i,j] or
                image[i, j-1] >= image[i,j]):
                thresholded[i, j] = 0
    return thresholded
  
def cornerDetector(im, sigma, epsilon, k, tau):
    r = harrisMeasure(im, sigma, epsilon, k)
    r[r < tau] = 0
    thr = threshold_neighbors(r)
    thr[thr > tau] = 1
    c = np.where(thr==1)
    t_y = c[0]
    t_x = c[1]
    return [t_x,t_y]

def test_points(n_in, n_out):
    a = (np.random.rand(n_in)-.5)*10
    b = np.vstack((a, a*.5+np.random.randn(n_in)*.25))
    points = np.hstack((b, 2*np.random.randn(2, n_out)))
    return np.random.permutation(points.T).T
def pp_line(p1,p2):
    l = np.cross(p1.T,p2.T) 
    l = np.reshape(l,(3,))
    l = l/(np.sqrt(l[0]**2+l[1]**2))
    return l

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
    im_scales = scaleSpaced(im, sigma, n)


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

def Fest_8point(q1,q2):
    
    B = []
    N = q1.shape[1]
    
    for i in range(N):
        x1 = q1[0,i]
        y1 = q1[1,i]  
        x2 = q2[0,i]
        y2 = q2[1,i] 
        
        B.append(np.array([x1*x2,y1*x2,x2,x1*y2,y1*y2,y2,x1,y1,1]))
    
    B = np.array(B)
    
    U, S, V = np.linalg.svd(B)
    F = V[-1, :].reshape((3, 3))
    return F

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

def RANSAC2(points, N=100, threshold=0.1, p=0.99):
    best_line = None
    best_consensus = 0
    m = points.shape[1]
    s = 2

    for n in range(N):

        e_hat = 1 - s/m

        N_hat = np.log(1-p)/np.log((1-(1-e_hat)*2))

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

def SIFT_feat():
    
    im1 = cv2.imread("data/im1.jpg")    
    im2 = cv2.imread("data/im2.jpg")
    
    # Create a SIFT object
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors for both images
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)
    
    # Create a brute-force matcher object with cross-checking
    bf = cv2.BFMatcher(crossCheck=True)
    
    # Match descriptors of keypoints in both images
    matches = bf.match(des1, des2)
    
    # Filter out good matches based on a distance threshold
    good = []
    for m in matches:
        if m.distance < 0.1 * min(len(kp1), len(kp2)):
            good.append(m)
    
    # Draw the matched keypoints on the images
    img_matches = cv2.drawMatches(im1, kp1, im2, kp2, good, None)
    return img_matches

