# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:24:52 2023

@author: danie
"""

import cv2
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import scipy

def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

def Pi(point):
    return point[:-1]/point[-1]

def PiInv(point):
   ones_row = np.ones((1, point.shape[1]))
   return np.concatenate((point, ones_row), axis=0)

def projectpoints(K, R, t, Q):
    Rt = np.concatenate((R,t), axis=1)
    p_h = K@Rt@PiInv(Q)
    p = Pi(p_h)
    return p

def distprojectpoints(K, R, t, Q, k3, k5, k7):
    Rt = np.concatenate((R,t), axis=1)
    p = Pi(Rt@PiInv(Q))
    x = p[0, :]
    y = p[1, :]
    r = np.sqrt(x**2 + y**2)
    dist = (1 + k3 * r**2 + k5 * r**4 + k7 * r**6)
    up = np.zeros_like(p)
    up[0, :] = x * dist
    up[1, :] = y * dist
    p = K@PiInv(up)
    
    return p

def undistortImage(im, K, k3, k5, k7):
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)
    q = Pi(np.linalg.inv(K)@p)
    x1 = q[0, :]
    y1 = q[1, :]
    r = np.sqrt(x1**2 + y1**2)
    dist = (1 + k3 * r**2 + k5 * r**4 + k7 * r**6)
    uq = np.zeros_like(q)
    uq[0, :] = x1 * dist
    uq[1, :] = y1 * dist
    q_d = uq
    p_d = K@PiInv(q_d)
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    assert (p_d[2]==1).all(), 'You did a mistake somewhere'
    im_undistorted = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)
    return im_undistorted

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

def CrossOp(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])

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

def checkerboard_points(n,m):
    i, j = np.meshgrid(range(n), range(m), indexing='ij')

    center_i = (n-1)/2.0
    center_j = (m-1)/2.0

    Q = np.vstack((i.flatten() - center_i, j.flatten() - center_j, np.zeros(n*m)))
    
    return Q

def estimateHomographies(Q_omega, qs):
    H = []
        
    for i in range(len(qs)):
        q = qs[i]
                
        H.append(hest(Q_omega[:2, :], q))
    
    return H

def estimate_b(Hs):
    V = np.zeros((2 * len(Hs), 6))
    for i in range(len(Hs)):
        H = Hs[i]
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
    
    return b

def estimateIntrinsics(Hs):
    
    b = estimate_b(Hs)

    v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)
    
    lambda_ = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0]
    alpha = np.sqrt(lambda_ / b[0])
    
    beta = np.sqrt(lambda_*b[0] / (b[0]*b[2] - b[1]**2))
    gamma = -1.0 * b[1] * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - b[3] * alpha**2 / lambda_

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,     1]])
    return K

def estimateExtrinsics(K, Hs):
    Rs = []
    ts = []
    for i in range(len(Hs)):
        H = Hs[i]
        
        lambda_ = 1/np.linalg.norm(np.linalg.inv(K)@H[:,0])
        
        r1 = lambda_*np.linalg.inv(K)@H[:,0]
        r2 = lambda_*np.linalg.inv(K)@H[:,1]
        r3 = np.cross(r1, r2)
        
        #R = np.array([r1, r2, r3])
        R = np.column_stack((r1, r2, r3))
        
        t = lambda_*np.linalg.inv(K)@H[:,2]
        Rs.append(R)
        ts.append(t)
    
    if ts[0][2] < 0:
        Rs = []
        ts = []
        for i in range(len(Hs)):
            H = -Hs[i]
        
            lambda_ = 1/np.linalg.norm(np.linalg.inv(K)@H[:,0])
        
            r1 = lambda_*np.linalg.inv(K)@H[:,0]
            r2 = lambda_*np.linalg.inv(K)@H[:,1]
            r3 = np.cross(r1, r2)
        
        #R = np.array([r1, r2, r3])
            R = np.column_stack((r1, r2, r3))
        
            t = lambda_*np.linalg.inv(K)@H[:,2]
            Rs.append(R)
            ts.append(t)
    
    return Rs, ts

def calibratecamera(qs, Q):
    Hs = estimateHomographies(Q, qs)
    K = estimateIntrinsics(Hs)
    Rs, ts = estimateExtrinsics(K, Hs)
    return K, Rs, ts

def triangulate_nonlin(q_list, P_list):
    
    Q = triangulate(q_list, P_list)
    
    def compute_residuals(Q):
        res = []
        Q = Q.reshape(3,1)
        for i in range(len(P_list)):
            q = q_list[i]
            P = P_list[i]
            r = (Pi(P@PiInv(Q)) - q)
            res.append(r)
        return np.array(res).reshape(len(P_list)*2,)
    
    result = scipy.optimize.least_squares(compute_residuals, Q)["x"]
    
    return result

def gaussian1DKernel(sigma, size=4):
    s = np.ceil(np.max([sigma*size, size]))
    x = np.arange(-s,s+1)
    x = x.reshape(x.shape + (1,))
    g = np.exp(-x**2/(2*sigma*sigma))
    g /= np.sum(g)
    gd = -x/(sigma*sigma)*g
    return g, gd

def gaussianSmoothing(im, sigma):
    g,gd = gaussian1DKernel(sigma, size=4)
    im = scipy.ndimage.convolve(im,g.T)
    I = scipy.ndimage.convolve(im,g)
    Iy = scipy.ndimage.convolve(im,gd)
    Ix = scipy.ndimage.convolve(im,gd.T)
    return I, Ix, Iy


def smoothedHessian(im, sigma, epsilon):
    g_ep, g_epd = gaussian1DKernel(sigma, size=4)
    g_ep2 = np.outer(g_ep,g_ep)
    I, Ix, Iy = gaussianSmoothing(im, sigma)
    c11 = scipy.ndimage.convolve(Ix**2, g_ep2)
    c12 = scipy.ndimage.convolve(Ix*Iy, g_ep2)
    c21 = scipy.ndimage.convolve(Ix*Iy, g_ep2)
    c22 = scipy.ndimage.convolve(Iy**2, g_ep2)
    C = [[c11, c12], [c21, c22]]
    return C

def harrisMeasure(im, sigma, epsilon, k):
    C = smoothedHessian(im, sigma, epsilon)
    a = C[0][0]
    c = C[0][1]
    b = C[1][1]
    r = a*b-c**2-k*(a+b)**2
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
    #thr[thr > tau] = 1
    c = np.where(thr!=0)
    tempy = c[0]
    tempx = c[1]
    c = [tempx, tempy]
    return c
    
def line(p1, p2):
    p1 = PiInv(p1)
    p2 = PiInv(p2)
    l = np.cross(p1.T,p2.T)
    l = np.reshape(l, (3,))
    l = l / (np.sqrt(l[0]**2+l[1]**2))
    return l

def in_out(points, l, threshold):
    # dist = formel for distance mellem linje og punkt (fra Wikipedia)
    dist = np.abs(l[0]*points[0]+l[1]*points[1]+l[2]) / np.sqrt(l[0]**2+l[1]**2)
    inliers = dist < threshold
    return inliers

def consensus(points, l, threshold):
    inout = in_out(points, l, threshold)
    consensus = np.sum(inout)
    return consensus

def draw_two_points(points):
    N = points.shape[1]
    
    indices = np.random.choice(range(N), size=2, replace=False)
    
    p1 = points[:, indices[0]]
    p2 = points[:, indices[1]]
    
    return p1, p2

