import numpy as np

def ComputeEssential(K, F):
    e = K.T.dot(F).dot(K)
    U,s,V = np.linalg.svd(e)
    s = [1,1,0]
    E = np.dot(U,np.dot(np.diag(s),V))
    return E