import numpy as np
import matplotlib.pyplot as plt
import h5py
from parameters import *


class Data:

    def __init__(self, data_filename):


        d = h5py.File(data_filename)

        self.input_data = np.array(d['results']['vars'])
        self.days_since_prev_entry = np.array(d['results']['days_since_prev_entry'])
        self.future_max_vert = np.array(d['results']['future_max_vert'])
        self.wheel_index_range = np.array(d['results']['wheel_index_range'],dtype=np.int32)
        self.ID =  np.array(d['results']['ID'],dtype=np.int32)
        self.ID_count =  np.array(d['results']['ID_count'],dtype=np.int32)
        self.valid_last_data_pts = np.array(d['results']['valid_last_data_pts'])
        self.valid_last_data_pts = np.where(self.valid_last_data_pts>0)[1]
        d.close()

        self.num_entries = self.input_data.shape[1]
        self.num_valid_last_data_pts = len(self.valid_last_data_pts)
        self.num_vars = self.input_data.shape[0]

        # remove nans and z-score data
        u = np.tile(np.nanmean(self.input_data, axis=1, keepdims=True), (1,self.input_data.shape[1]))
        sd = np.tile(np.nanstd(self.input_data, axis=1, keepdims=True), (1,self.input_data.shape[1]))
        for d in range(self.num_vars):
            ind = np.where(np.isnan(self.input_data[d,:]))[0]
            self.input_data[d,ind] = u[d,0]
        self.input_data = (self.input_data-u)/sd

        # cap at +/-6 SDs
        self.input_data = np.maximum(-6, np.minimum(6,self.input_data))

        print('Number vars = ', self.num_vars)



    def generate_batch_data(self, test_data = False):

        # if we're testing data, will generate multiple reps for every entry
        if test_data:
            num_reps = par['test_reps']
        else:
            num_reps = 1

        x = np.zeros((num_reps, par['batch_size'], (self.num_vars+2)*par['hist_size']-2), dtype = np.float32)
        y = np.zeros((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)

        for i in range(par['batch_size']):
            # randomly select batch_size entries
            ind = self.valid_last_data_pts[np.random.randint(self.num_valid_last_data_pts)]
            y[i,0] = self.future_max_vert[1,ind]
            ind_range = range(self.wheel_index_range[0,ind], self.wheel_index_range[1,ind])
            ind_preceeding = self.wheel_index_range[0,ind] + np.where(self.ID_count[0,ind_range] < self.ID_count[0,ind])[0]

            for r in range(num_reps):

                ind_hist = np.random.permutation(len(ind_preceeding))
                ind_hist = ind_preceeding[ind_hist[:par['hist_size']-1]]
                ind_hist = np.sort(ind_hist)

                u = range(self.num_vars)
                x[r,i,u] = self.input_data[:, ind]
                for j in range(par['hist_size']-1):
                    # wheel variables
                    u = range(self.num_vars*(j+1),self.num_vars*(j+2))
                    x[r,i,u] = self.input_data[:, ind_hist[j]]

                    # days since prev entry
                    u1 = self.num_vars*par['hist_size']+j
                    u2 = self.num_vars*par['hist_size']+par['hist_size']-1+j
                    #  TESTING!
                    x[r,i,u1] = 2*np.exp(-self.days_since_prev_entry[0,ind_hist[j]]/7)
                    x[r,i,u2] = 2*np.exp(-self.days_since_prev_entry[0,ind_hist[j]]/30)

        #plt.imshow(np.squeeze(x), aspect='auto', interpolation = 'none')
        #plt.colorbar()
        #plt.show()
        return np.squeeze(x), y
