



import data_loader
import rbm
import sgd

def run():

    # load training data
    trainImages,trainLabels=data_loader.load_mnist_train()

    print "Done Loading Data"
    # height of image
    imDim = trainImages.shape[0]

    visibleSize = imDim**2
    hiddenSize = 300
    
    # reshape images into vector
    trainImages = trainImages.reshape(visibleSize,-1)

    RBM = rbm.RBM(visibleSize,hiddenSize,sp_target=0.1,sp_weight=.5)

    # initialize RBM parameters
    RBM.initParams()

    SGD = sgd.SGD(RBM,epochs=2,alpha=1e-3,minibatch=50)

    # run SGD loop
    SGD.run(trainImages)

    # view learned features post training
    RBM.view_weights(imDim)

    # run 10 chains with gibbs sampling on trained model

    samples = RBM.gibbs_sample(10,trainImages[:,:10])

    samples = samples.reshape(imDim,imDim,-1)

    RBM.view_patches(samples,samples.shape[2])
    
    

if __name__=='__main__':
    run()
