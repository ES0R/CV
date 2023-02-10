import numpy as np

def Pi(ph):
    p = ph[:-1]/ph[-1]
    return p

def PiInv(p):
    H = np.shape(p)[1]
    homo = np.ones((1, H))
    ph = np.concatenate((p,homo),axis = 0)
    return ph

def projectpoints(K,R,t,Q):
    
    t = np.transpose([t])
    H = np.concatenate((R,t),axis=1)
    Q = PiInv(Q)

    return K@H@Q

def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2
