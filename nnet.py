
import numpy as np

class NNet:
    def __init__(self,inputDim,outputDim,layerSizes):
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.layerSizes = layerSizes
        self.weights = None

    def initializeWeights(self):
        sizesIn = [self.inputDim] + self.layerSizes
        sizesOut = self.layerSizes + [self.outputDim]
        self.weights = [[np.sqrt(6)/np.sqrt(n+m)*np.random.randn(m,n),np.zeros((n,1))] \
                            for n,m in zip(sizesIn,sizesOut)]
            
        
