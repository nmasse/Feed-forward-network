import numpy as np
import matplotlib.pyplot as plt
import h5py
from parameters import *


class Data:

    def __init__(self, data_filename):

        #f = 'C:/Users/nicol_000/Projects/Corey Train Analysis/IRRIS_Raw_2014-2016/good code/data_odd.mat'
        #d = tables.open_file(data_filename)

        d = h5py.File(data_filename)
        [print(p) for p in d['results'].keys()]
        print(d['results']['vars'])
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
        self.input_data[np.isnan(self.input_data)] = 0
        u = np.tile(np.mean(self.input_data, axis=1, keepdims=True), self.input_data.shape[1])
        sd = np.tile(np.std(self.input_data, axis=1, keepdims=True), self.input_data.shape[1])
        self.input_data = (self.input_data-u)/sd

        print('Number vars = ', self.num_vars)



    def generate_batch_data(self):

        # randomly select batch_size entries
        x = np.zeros((par['batch_size'], (self.num_vars+1)*par['hist_size']-1), dtype = np.float32)
        y = np.zeros((par['batch_size'], 2), dtype = np.float32)


        for i in range(par['batch_size']):
            ind = self.valid_last_data_pts[np.random.randint(self.num_valid_last_data_pts)]
            ind_range = range(self.wheel_index_range[0,ind], self.wheel_index_range[1,ind])
            ind_preceeding = self.wheel_index_range[0,ind] + np.where(self.ID_count[0,ind_range] < self.ID_count[0,ind])[0]
            ind_hist = np.random.permutation(len(ind_preceeding))
            ind_hist = ind_preceeding[ind_hist[:par['hist_size']-1]]
            ind_hist = np.sort(ind_hist)

            y[i,:] = self.future_max_vert[:,ind]
            u = range(self.num_vars)
            x[i,u] = self.input_data[:, ind]
            for j in range(par['hist_size']-1):
                # wheel variables
                u = range(self.num_vars*(j+1),self.num_vars*(j+2))
                x[i,u] = self.input_data[:, ind_hist[j]]

                # days since prev entry
                u = self.num_vars*par['hist_size']+j
                x[i,u] = self.days_since_prev_entry[0,ind_hist[j]]

        #plt.imshow(x, aspect='auto', interpolation = 'none')
        #plt.show()
        return x, y
