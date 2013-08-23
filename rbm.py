
import numpy as np

    
class RBM:
    
    def __init__(self,visSize,hidSize):
        self.visSize = visSize
        self.hidSize = hidSize
        self.W = None
        self.c = None
        self.b = None

    def initParams(data=None):

        self.W = 0.01 * np.random.randn(self.hidSize,self.visSize)
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


            

