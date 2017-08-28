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
    'n_dendrites'           : 5,
    'init_weight_sd'        : 0.05,
    'learning_rate'         : 1e-3,
    'num_iterations'        : 100000,
    'iters_between_eval'    : 20,
    'batch_size'            : 1000,
    'n_perms'               : 10,
    'n_pixels'              : 784,
    'layer_dims'            : [784,120,120,120,120,10],
    'test_reps'             : 50,

    # Omega parameters
    'xi'                    : 0.001,

    # Dropout
    'keep_prob'             : 1
    }

def update_dependencies():
    """
    Updates all parameter dependencies
    """
    par['num_layers'] = len(par['layer_dims'])
    par['n_hidden_layers'] = par['num_layers'] - 2

update_dependencies()

print("--> Parameters successfully loaded.\n")
