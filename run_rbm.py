



import data_loader
import rbm
import sgd
import visualizer as vsl



def run():

    # load training data
    trainImages,trainLabels=data_loader.load_mnist_train()

    print "Done loading data..."
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

    print "Training..."
    # run SGD loop
    SGD.run(trainImages)

    # view up to 100 learned features post training
    vsl.view_patches(W.reshape(imDim,imDim,hiddenSize),min(hiddenSize,100))

    print "Sampling Gibbs chains..."
    # run 10 chains with gibbs sampling on trained model
    samples = RBM.gibbs_sample(10,trainImages[:,:10])

    # view samples from gibbs chains
    samples = samples.reshape(imDim,imDim,-1)
    vsl.view_patches(samples,samples.shape[2])

if __name__=='__main__':
    run()
