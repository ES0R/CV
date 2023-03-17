
import numpy as np

def test():
    print("test")
    return

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
