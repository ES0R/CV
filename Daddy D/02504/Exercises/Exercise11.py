# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:58:18 2023

@author: danie
"""


#%% IMPORTS
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions import triangulate_nonlin
#%% 11.1

K = np.loadtxt('Glyp/K.txt')


im0 = cv2.imread("Glyp/sequence/000001.png")
im1 = cv2.imread("Glyp/sequence/000002.png")
im2 = cv2.imread("Glyp/sequence/000003.png")

im0 = im0[:,:,::-1]
im1 = im1[:,:,::-1]
im2 = im2[:,:,::-1]

sift = cv2.SIFT_create(nfeatures=1999)

kp0, des0 = sift.detectAndCompute(im0, None)
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)


bf = cv2.BFMatcher(crossCheck=True)

# Match descriptors of keypoints in both images
matches01 = bf.match(des0, des1)
matches12 = bf.match(des1, des2)

matches01 = sorted(matches01, key = lambda x:x.distance)
matches12 = sorted(matches12, key = lambda x:x.distance)


matches01 = np.array([(m.queryIdx, m.trainIdx) for m in matches01])
matches12 = np.array([(m.queryIdx, m.trainIdx) for m in matches12])



#%% 11.2

kp0t = [kp0[i] for i in matches01[:,0]]
kp1t = [kp1[i] for i in matches01[:,0]]
kp1t2 = [kp1[i] for i in matches12[:,0]]
kp2t2 = [kp2[i] for i in matches12[:,0]]



kp0t = np.array([k.pt for k in kp0t])
kp1t = np.array([k.pt for k in kp1t])
kp1t2 = np.array([k.pt for k in kp1t2])
kp2t2 = np.array([k.pt for k in kp2t2])


E1, mask1 = cv2.findEssentialMat(kp0t, kp1t , K)
E2, mask2 = cv2.findEssentialMat(kp1t2, kp2t2 , K)


points1, R1, t1, mask1 = cv2.recoverPose(E1, kp0t, kp1t)
points2, R2, t2, mask2 = cv2.recoverPose(E2, kp1t2, kp2t2)

matches01_inliers = matches01[mask1.ravel() == 255]
matches12_inliers = matches12[mask2.ravel() == 255]



#%% 11.3

_, idx01, idx12 = np.intersect1d(matches01_inliers[:,1], matches12_inliers[:,0], return_indices=True)


matches01_subset = matches01_inliers[idx01,:]
matches12_subset = matches12_inliers[idx12,:]


#%%


points0 = np.array([kp0[m[0]].pt for m in matches01_subset])
points1 = np.array([kp1[m[1]].pt for m in matches01_subset])
points2 = np.array([kp2[m[1]].pt for m in matches12_subset])

fig, (ax0, ax1) = plt.subplots(ncols=2)
        
# Display the two images on the subplots
ax0.imshow(im0)
ax1.imshow(im1)

# Plot the two points on top of the respective images

i = 12

ax0.plot(points0[i][0], points0[i][1], 'ro')  # 'ro' means red circle
ax1.plot(points1[i][0], points1[i][1], 'bo')  # 'bo' means blue circle

# Show the figure
plt.show()



#%%


#points_list0 = [np.array(point).reshape(2, 1) for point in points0]

Rt0 = np.concatenate((R1,t1), axis=1)

P0 = K@Rt0

Rt1 = np.concatenate((R2,t2), axis=1)

P1 = K@Rt1

P_list = [P0,P1]


Q = []

for i in range(points0.shape[0]):
    
    points_list0 = [points0[i,:].reshape(2,1),points1[i,:].reshape(2,1)]
    Q.append(triangulate_nonlin(points_list0, P_list))

Q = np.array(Q)

[rvec, tvec, success, inliers] = cv2.solvePnPRansac(Q,points2, K,distCoeffs = np.zeros(5))

#%%
Q_in = Q[inliers].reshape(inliers.shape[0],3)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Q_in[:,0],Q_in[:,1],Q_in[:,2])




















