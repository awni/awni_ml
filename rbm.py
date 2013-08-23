
import numpy as np

# sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sample_binomial(probs):
    samples = probs.copy()
    mask = probs>0.5
    sampless[mask] = 1
    samples[!mask] = 0
    return samples
    
    
class RBM:
    
    def __init__(self,visSize,hidSize):
        self.visSize = visSize
        self.hidSize = hidSize
        self.W = None
        self.c = None
        self.b = None

    def initParams(self,data=None):

        self.W = 0.01 * np.random.randn(self.visSize,self.hidSize)
        self.c = np.zeros((self.hidSize,1))
        
        if data = None:
            self.b = np.zeros((self.visSize,1))
        else:
            # advice from Hinton (practical guide to training RBM)
            # init visible bias to log(p_i/(1-p_i)) where p_i is
            # fraction of training set that is on for ith unit
            pVis = np.sum(data>0.5,axis=1)
            pVis = pVis/data.shape[1]
            self.b = np.log(pVis/(1-pVis))

    # updates params from unrolled vector
    def setParams(self,theta):
        self.W = theta[:self.W.size].reshape(self.visSize,self.hidSize)
        self.b = theta[self.W.size:self.W.size+self.b.size].reshape(-1,1)
        self.c = theta[self.W.size+self.b.size:].reshape(-1,1)

    # sample the visible units
    def sample_v(self,h):
        probs = sigmoid(self.b+self.W.dot(h))
        return probs

    # sample the hidden units
    def sample_h(self,v):
        probs = sigmoid(self.c+self.W.T.dot(v))
        return probs


    # approximate gradient with CD-1 updates
    def grad(data):

        p_h1 = self.sample_h(data)
        h1 = sample_binomial(p_h1)
        p_v2 = self.sample_v(h1)
        p_h2 = self.sample_h(p_v2)
        
        Wgrad = data.dot(p_h1.T) - p_v2.dot(p_h2.T)
        bgrad = np.sum(p_h1,axis=1) - np.sum(p_h2,axis=1)
        cgrad = np.sum(data,axis=1) - np.sum(p_v2,axis=1)

        # unroll gradient for optimizer
        grad = np.hstack((Wgrad.ravel(),bgrad.ravel(),cgrad.ravel()))



    
