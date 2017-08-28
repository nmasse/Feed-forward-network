import numpy as np
from mnist import MNIST
from parameters import *
import itertools

class Data:

    def __init__(self):

        # Load MNIST data
        mndata = MNIST('./resources/mnist/data/original')
        self.train_images, self.train_labels = mndata.load_training()
        self.test_images, self.test_labels   = mndata.load_testing()

        print('Num. available train images:', len(self.train_images))
        print('Num. available test images: ', len(self.test_images))

        # Generate MNIST image permutations
        self.permutations = np.zeros((par['n_perms'], par['n_pixels']) ,dtype=np.int16)
        for i in range(1, par['n_perms']):
            self.permutations[i, :] = np.random.permutation(par['n_pixels'])


    def generate_batch_data(self, perm_ind=0, test_data=False):

        # Allocate x and y arrays
        x = np.zeros((par['n_pixels'], par['batch_size']), dtype = np.float32)
        y = np.zeros((10, par['batch_size']), dtype = np.float32)

        # Generate appropriate indices for batch data
        if test_data:
            ind = np.random.permutation(len(self.test_labels))
        else:
            ind = np.random.permutation(len(self.train_labels))
        ind = ind[:par['batch_size']]

        for (i, img), (j, p) in itertools.product(enumerate(ind), enumerate(self.permutations[perm_ind,:])):
            if test_data:
                x[j,i] = self.test_images[img][p]/255
                y[self.test_labels[img], i] = 1
            else:
                x[j,i] = self.train_images[img][p]/255
                y[self.train_labels[img], i] = 1

        return x, y
