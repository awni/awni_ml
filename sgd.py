

import numpy as np


class SGD:

    def __init__(self,model,momentum=0.9,epochs=1,alpha=1e-2,
                 minibatch=256):
        
        self.model = model

        assert self.model is not None, "Must define a function to optimize"

        self.momentum = momentum # momentum
        self.epochs = epochs # number of epochs through the data
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch

    # runs stochastic gradient descent with model as objective.  Expects
    # data in n x m matrix where n is feature dimension and m is numbe of
    # training examples
    def run(self,data,labels=None):
        
        m = data.shape[1]
        
        # momentum setup
        velocity = np.zeros(self.model.numParams)
        momIncrease = 500
        mom = 0.5

        it = 0
        for e in xrange(self.epochs):
            # randomly select minibatch
            perm = np.random.permutation(range(m))

            for i in range(0,m-self.minibatch+1,self.minibatch):
                it += 1

                mb_data = data[:,perm[i:i+self.minibatch]]

                if labels is None:
                    cost,grad = self.model.grad(mb_data)
                else:
                    mb_labels = labels[perm[i:i+self.minibatch]]
                    cost,grad = self.model.grad(mb_data,mb_labels)

                if it > momIncrease:
                    mom = self.momentum

                # update velocity
                velocity = mom*velocity+self.alpha*np.squeeze(grad)
                
                # update params
                self.model.updateParams(-velocity)
                
                if it%100 == 0:
                    print "Reconstruction cost on iteration %d is %f."%(it,cost)
            
            print "Done with epoch %d."%(e+1)
            # anneal learning rate by factor of 2 after each epoch
            self.alpha = self.alpha/2.0
            
