import numpy as np
import os

print("--> Loading parameters...")

global par

"""
Independent parameters
"""
par = {
    # Setup parameters
    'save_dir'              : './save_dir/',
    'data_dir'              : './data_dir/',
    'data_filenames'        : ['data_even.mat', 'data_odd.mat'],
    'debug_model'           : False,
    'load_previous_model'   : False,
    'ckpt_load_fn'          : 'model.ckpt',
    'ckpt_save_fn'          : 'model.ckpt',

    # Network configuration
    'layer_dims'            : [79,200,150,100,50,2],
    'nonlinearity'          : 'sigmoid',
    'learning_rate'         : 1e-3,
    'num_iterations'        : 1000,
    'batch_size'            : 250,
    'hist_size'             : 10
    }

def update_dependencies():
    """
    Updates all parameter dependencies
    """
    par['num_layers'] = len(par['layer_dims'])

update_dependencies()

print("--> Parameters successfully loaded.\n")
