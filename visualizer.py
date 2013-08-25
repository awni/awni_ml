
import matplotlib.pyplot as plt
import numpy as np

# helper function for plotting patches to an image
def view_patches(patches,num):

    xnum = int(np.sqrt(num))
    if xnum**2 == num:
        ynum = xnum
    else:
        ynum = xnum+1

    patchDim = patches.shape[0]
    if np.min(patches) < 0:
        image = -np.ones(((patchDim+1)*ynum,(patchDim+1)*xnum))
    else:
        image = np.zeros(((patchDim+1)*ynum,(patchDim+1)*xnum))
    for i in range(ynum):
        for j in range(xnum):
            imnum = i*xnum+j
            if imnum>=num:
                break
            image[i*(patchDim+1):i*(patchDim+1)+patchDim, \
                  j*(patchDim+1):j*(patchDim+1)+patchDim] \
                  = patches[:,:,imnum]
    plt.imshow(image,cmap=plt.get_cmap('gray'))
    plt.show()
