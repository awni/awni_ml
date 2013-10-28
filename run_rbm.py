
import data_loader
import rbm
import sgd
import visualizer as vsl
import preprocess


def run():

    print "Loading data..."
    # load training data
    trainImages,trainLabels=data_loader.load_mnist_train()


    imDim = trainImages.shape[0]

    visibleSize = 784 # 69
    hiddenSize = 500

    trainImages = trainImages.reshape(imDim**2,-1)
    import numpy as np
    trainImages = trainImages - np.mean(trainImages,axis=1).reshape(-1,1)

    # preprocess
    print "Preprocessing Data..."
    #prp = preprocess.Preprocess()
    #prp.computePCA(trainImages)
    # prp.plot_explained_var()
    #trainImages = prp.whiten(trainImages,numComponents=visibleSize)

    RBM = rbm.RBM(visibleSize,hiddenSize,grbm=True,sp_target=0.05,sp_weight=5)

    # initialize RBM parameters
    RBM.initParams()

    SGD = sgd.SGD(RBM,epochs=2,alpha=1e-5,minibatch=50)


    # run SGD loop
    print "Training..."
    SGD.run(trainImages)

    # view up to 100 learned features post training
    W = RBM.W #prp.unwhiten(RBM.W)
    vsl.view_patches(W.reshape(imDim,imDim,hiddenSize),min(hiddenSize,100))

    print "Sampling Gibbs chains..."
    # run 10 chains with gibbs sampling on trained model
    #samples = RBM.gibbs_sample(10,trainImages[:,:10])

    # view samples from gibbs chains
    #samples = prp.unwhiten(samples)
    #samples = samples.reshape(imDim,imDim,-1)
    #vsl.view_patches(samples,samples.shape[2])

if __name__=='__main__':
    run()
