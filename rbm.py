
import numpy as np

# sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sample_binomial(probs):
    samples = probs.copy()
    mask = probs>0.5
    samples[mask] = 1
    samples[~mask] = 0
    return samples
    
    
class RBM:
    
    def __init__(self,visSize,hidSize):
        self.visSize = visSize
        self.hidSize = hidSize
        self.W = None
        self.c = None
        self.b = None
        self.numParams = visSize*hidSize+visSize+hidSize

    def initParams(self,data=None):

        self.W = 0.01 * np.random.randn(self.visSize,self.hidSize)
        self.c = np.zeros((self.hidSize,1))
        
        if data is None:
            self.b = np.zeros((self.visSize,1))
        else:
            # advice from Hinton (practical guide to training RBM)
            # init visible bias to log(p_i/(1-p_i)) where p_i is
            # fraction of training set that is on for ith unit
            pVis = np.sum(data>0.5,axis=1)
            pVis = pVis/float(data.shape[1])
            minAct = np.min(pVis[pVis>0])
            pVis[pVis==0] = minAct
            self.b = np.log(pVis/(1-pVis))
            self.b = self.b.reshape(-1,1)

    # updates params from unrolled vector
    def updateParams(self,update):
        self.W = self.W + update[:self.W.size].reshape(self.visSize,self.hidSize)
        self.b = self.b + update[self.W.size:self.W.size+self.b.size].reshape(-1,1)
        self.c = self.c + update[self.W.size+self.b.size:].reshape(-1,1)

    # sample the visible units
    def sample_v(self,h):
        probs = sigmoid(self.b+self.W.dot(h))
        return probs

    # sample the hidden units
    def sample_h(self,v):
        probs = sigmoid(self.c+self.W.T.dot(v))
        return probs


    # approximate gradient with CD-1 updates
    def grad(self,data):

        # sampling
        p_h1 = self.sample_h(data)
        h1 = sample_binomial(p_h1)
        p_v2 = self.sample_v(h1)
        #import sys
        #sys.exit(0)

        p_h2 = self.sample_h(p_v2)

        # reconstruction cost
        cost = np.sqrt(np.sum((data-p_v2)**2))

        Wgrad = p_v2.dot(p_h2.T) - data.dot(p_h1.T)
        bgrad = np.sum(p_h2,axis=1) - np.sum(p_h1,axis=1) 
        cgrad = np.sum(p_v2,axis=1) - np.sum(data,axis=1)

        # unroll gradient for optimizer
        grad = np.hstack((Wgrad.ravel(),bgrad.ravel(),cgrad.ravel()))

        return cost,grad


    
