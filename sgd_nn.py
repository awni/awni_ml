"""
 Author: Awni Hannun
 Class: SGD - Stochastic Gradient Descent Optimizer
 Runs stochastic gradient descent with momentum.
 Options:
    model    :  model/objective to be optimized.  Must support
     the following methods- model.grad returns the unrolled 
     gradient with respect to every parameter, model.updateParams 
     accepts an update vector of deltas and updates all the 
     parameters of the model, model.numParams instance variable 
     for number of parameters in model.
    momentum  :  momentum to use
    epochs    :  number of epochs over full dataset
    alpha     :  learning rate
    minibatch :  size of minibatch
"""
import numpy as np
import gnumpy as gp

class SGD:

    def __init__(self,model,momentum=0.9,epochs=1,alpha=1e-2,
                 minibatch=256,gpu=False):
        
        self.model = model

        assert self.model is not None, "Must define a function to optimize"

        self.momentum = momentum # momentum
        self.epochs = epochs # number of epochs through the data
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch
        self.gpu = gpu # use gpu?

    def run(self,data,labels=None):
        """
        Runs stochastic gradient descent with model as objective.  Expects
        data in n x m matrix where n is feature dimension and m is numbe of
        training examples
        """
        m = data.shape[1]
        
        # momentum setup
        #velocity = np.zeros(self.model.numParams)
        momIncrease = 500
        mom = 0.5

        it = 0
        for e in xrange(self.epochs):
            # randomly select minibatch
            perm = np.random.permutation(range(m))

            for i in xrange(0,m-self.minibatch+1,self.minibatch):
                it += 1

                mb_data = data[:,perm[i:i+self.minibatch]]
                if self.gpu:
                    mb_data = gp.garray(mb_data)

                if labels is None:
                    cost,grad = self.model.costAndGrad(mb_data)
                else:
                    mb_labels = labels[perm[i:i+self.minibatch]]
                    cost,grad = self.model.costAndGrad(mb_data,mb_labels)

                if it > momIncrease:
                    mom = self.momentum

                # update velocity
                #velocity = mom*velocity+self.alpha*np.squeeze(grad)

                # update params
                self.model.updateParams(-self.alpha,grad)

                if it%10 == 0:
                    print "Cost on iteration %d is %f."%(it,cost)

            print "Done with epoch %d."%(e+1)
            
