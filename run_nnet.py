
import data_loader
import sgd_nn as sgd
import nnet
import gnumpy as gp
import numpy as np

def run():

    print "Loading data..."
    # load training data
    trainImages,trainLabels=data_loader.load_mnist_train()

    imDim = trainImages.shape[0]
    inputDim = imDim**2
    outputDim = 10
    layerSizes = [4048]*6

    trainImages = trainImages.reshape(inputDim,-1)

    mbSize = 256

    nn = nnet.NNet(inputDim,outputDim,layerSizes,mbSize)

    nn.initParams()

    epochs = 2
    
    SGD = sgd.SGD(nn,epochs=20,alpha=1e-2,minibatch=mbSize,gpu=True)

    # run SGD loop
    print "Training..."
    SGD.run(trainImages,trainLabels)

if __name__=='__main__':
    run()
