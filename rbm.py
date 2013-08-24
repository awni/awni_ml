# Author: Awni Hannun
# Class: RBM - Sparse (optional) Restricted Boltzmann Machine
# Options:
#   visSize   :  input dimension
#   hidSize   :  number of hidden units
#   sp_target :  target activation for hidden units 
#     (if nonzero RBM is trained with cross entropy 
#     sparsity penalty between average activation
#      and target activation probabilities)
#   sp_weight :  weight of sparsity penalty in parameter updates

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
    
    def __init__(self,visSize,hidSize,sp_target=0,sp_weight=0):
        self.visSize = visSize
        self.hidSize = hidSize
        self.W = None
        self.c = None
        self.b = None
        self.numParams = visSize*hidSize+visSize+hidSize
        self.p = sp_target  # target sparsity
        self.beta = sp_weight # weight of sparsity penalty
        self.lam = 0.9 # weight for averaging activation probas
        if self.p!=0:
            self.q = np.zeros(self.hidSize)
        
    def initParams(self,data=None):

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
            Wgrad += self.beta*data.dot(np.tile(del_sp.reshape(-1,1),[1,m]).T) 
            cgrad = cgrad + self.beta*del_sp
            
        # unroll gradient for optimizer
        grad = np.hstack((Wgrad.ravel(),bgrad.ravel(),cgrad.ravel()))

        return cost,grad

    def gibbs_sample(self,mixTime=100,numsamples=5):
        print "TODO"

    # view the first min(hiddenSize,100) features as grayscale images
    def view_weights(self,imDim):
        
        assert self.visSize%imDim == 0, \
            "image dimension must divide visible size"

        # reshape W into images
        W_view = self.W.reshape(imDim,imDim,self.hidSize)

        # view up to first 100 feats
        self.view_patches(W_view,min(self.hidSize,100))

    # helper function for plotting patches to an image
    def view_patches(self,patches,num):
        import matplotlib.pyplot as plt

        xnum = int(np.sqrt(num))
        if xnum**2 == num:
            ynum = xnum
        else:
            ynum = xnum+1
            
        patchDim = patches.shape[0]
        image = -np.ones(((patchDim+1)*xnum,(patchDim+1)*ynum))
        for i in range(ynum):
            for j in range(xnum):
                imnum = i*xnum+j
                if imnum>num:
                    break
                image[j*(patchDim+1):j*(patchDim+1)+patchDim, \
                      i*(patchDim+1):i*(patchDim+1)+patchDim] \
                      = patches[:,:,imnum]
        plt.imshow(image,cmap=plt.get_cmap('gray'))
        plt.show()
