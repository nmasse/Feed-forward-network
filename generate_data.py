import numpy as np
from mnist import MNIST
from parameters import *


class Data:

    def __init__(self):

        mndata = MNIST('C:/Users/nicol/Projects/GitHub/MNIST-dataset-in-different-formats/data/Original dataset/')
        #mndata = MNIST('C:\Users\nicol\Projects\GitHub\MNIST-dataset-in-different-formats\data\Original dataset')
        self.train_images, self.train_labels = mndata.load_training()
        self.test_images, self.test_labels = mndata.load_testing()

        print(len(self.train_images), len(self.test_images))

        # generate permutations
        self.permutations = np.zeros((par['n_perms'], par['n_pixels']) ,dtype=np.int16)
        for i in range(par['n_perms']):
            self.permutations[i, :] = np.random.permutation(par['n_pixels'])


    def generate_batch_data(self, perm_ind = 0, test_data = False):

        x = np.zeros((par['n_pixels'], par['batch_size']), dtype = np.float32)
        y = np.zeros((10, par['batch_size']), dtype = np.float32)

        if test_data:
            ind = np.random.permutation(len(self.test_labels))
        else:
            ind = np.random.permutation(len(self.train_labels))
        ind = ind[:par['batch_size']]

        for i, img in enumerate(ind):
            for j, p in enumerate(self.permutations[perm_ind,:]):
                if test_data:
                    x[j, i] = self.test_images[img][p]
                    y[self.test_labels[img], i] = 1
                else:
                    x[j, i] = self.train_images[img][p]
                    y[self.train_labels[img], i] = 1

        return x, y
