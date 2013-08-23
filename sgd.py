




class SGD:

    def __init__(self,funObj=None,momentum=0.9,epochs=1,alpha=1e-2):
        
        self.funObj = funObj

        assert self.funObj is not None, "Must define a function to optimize"

        self.momentum = momentum # momentum
        self.epochs = epochs # number of epochs through the data
        self.alpha = alpha # learning rate
        

    def run(self):
        

        it = 0
        for _ in xrange(epochs):
            
            cost,grad = funObj(mb_data)
            
            grad = 
