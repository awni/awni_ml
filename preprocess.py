
import numpy as np
import numpy.linalg as npl

# takes as input n x m matrix and returns a whitening matrix
def whiten(data):

    meanPatch = np.mean(data,axis=1)

    # subtract meanPatch
    tmpData = data-meanPatch
    
    # pixelwise covariance
    sigma = np.cov(data)

    # eigenvalue decomposition
    V,U = npl.eigh(sigma)
    
    # add a little to nonzero eigenvalues before inverting
    epsilon = 1e-4
    V = np.sqrt(1/(V+epsilon))
    
    # construct whitening matrix
    W = np.diag(V).dot(U)

    return W,meanPatch



