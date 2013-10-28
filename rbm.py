"""
 Author: Awni Hannun
 Class: RBM - Sparse (optional) Restricted Boltzmann Machine
 Options:
   visSize   :  input dimension
   hidSize   :  number of hidden units
   sp_target :  target activation for hidden units 
     (if nonzero RBM is trained with cross entropy 
     sparsity penalty between average activation
      and target activation probabilities)
   sp_weight :  weight of sparsity penalty in parameter updates
"""

import numpy as np

def sigmoid(z):
    """
    Sigmoid function
    """
    return 1/(1+np.exp(-z))

def gauss(z):
    """
    Gaussian with std=1 and mean=z
    """
    return np.random.standard_normal(size=z.shape)+z
    #return 1/np.sqrt(2*np.pi)*np.exp(-(z**2)/2)


def sample_bernoulli(probs):
    """
    Sample bernoulli distribution
    """
    samples = probs.copy()
    mask = probs>0.5
    samples[mask] = 1
    samples[~mask] = 0
    return samples
    
    
class RBM:
    
    def __init__(self,visSize,hidSize,sp_target=0,sp_weight=0,grbm=False):
        self.visSize = visSize
        self.hidSize = hidSize
        self.W = None
        self.c = None
        self.b = None
        self.numParams = visSize*hidSize+visSize+hidSize
        self.p = sp_target  # target sparsity
        self.beta = sp_weight # weight of sparsity penalty
        self.lam = 0.5 # weight for averaging activation probas
        if self.p!=0:
            self.q = np.zeros(self.hidSize)
        self.grbm = grbm
        
    def initParams(self,data=None):
        """
        Initialize parameters for RBM using unifrom based on
        fanin-fanout.
        """
        # initialize W uniformly on +/- 4*sqrt(6/(fanin+fanout))
        range = 4.0*np.sqrt(6.0/(self.visSize+self.hidSize))
        self.W = range*(2*np.random.rand(self.visSize,self.hidSize)-1)

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


    def updateParams(self,update):
        """
        updates params from unrolled vector
        """

        self.W = self.W + update[:self.W.size].reshape(self.visSize,self.hidSize)
        self.b = self.b + update[self.W.size:self.W.size+self.b.size].reshape(-1,1)
        self.c = self.c + update[self.W.size+self.b.size:].reshape(-1,1)


    def sample_v(self,h):
        """
        sample the visible units
        """

        if self.grbm==True:
            probs = gauss(self.b+self.W.dot(h))
        else:
            probs = sigmoid(self.b+self.W.dot(h))

        return probs


    def sample_h(self,v):
        """
        Sample the hidden units
        """
        probs = sigmoid(self.c+self.W.T.dot(v))
        return probs

    def costAndGrad(self,data):
        """
        approximate gradient with CD-1 updates
        """
        # sampling
        p_h1 = self.sample_h(data)
        h1 = sample_bernoulli(p_h1)
        p_v2 = self.sample_v(h1)
        p_h2 = self.sample_h(p_v2)

        # reconstruction cost
        cost = np.sqrt(np.sum((data-p_v2)**2))
            
        Wgrad = p_v2.dot(p_h2.T) - data.dot(p_h1.T)
        bgrad = np.sum(p_v2,axis=1) - np.sum(data,axis=1) 
        cgrad = np.sum(p_h2,axis=1) - np.sum(p_h1,axis=1)

        # calculate sparsity penalty and gradient
        if self.p != 0:
            m = data.shape[1]
            # estimated activation probabilities for each unit
            self.q = self.lam*self.q+(1-self.lam)*(1.0/m)*np.sum(p_h1,axis=1)

            # sparsity grad
            del_sp = self.q - self.p

            # add in gradient from sparsity penalty
            #Wgrad += self.beta*data.dot(np.tile(del_sp.reshape(-1,1),[1,m]).T) 
            cgrad = cgrad + self.beta*del_sp
            
        # unroll gradient for optimizer
        grad = np.hstack((Wgrad.ravel(),bgrad.ravel(),cgrad.ravel()))

        return cost,grad

    def gibbs_sample(self,numSamples,data):
        """
        Collect numSamples samples starting with data as initialization
        every 1000 iterations of gibbs sampling.  Number of examples in 
        data will be number of chains.
        """

        numChains = data.shape[1]

        samples = np.empty((self.visSize,numChains*numSamples))

        v = data
        step = 1000 # collect sample after step iterations
        for i in range(numSamples):
            for _ in range(step):
                p_h = self.sample_h(v)  # sample hidden
                h = sample_bernoulli(p_h)
                v = self.sample_v(h) # sample visible
            samples[:,numChains*i:numChains*(i+1)] = v

        return samples
