import numpy as np
import tensorflow as tf
import os
import itertools

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
    'optimizer'             : 'MSE',           # MSE or cross_entropy
    'n_dendrites'           : 1,
    'init_weight_sd'        : 0.05,
    'learning_rate'         : 5e-3,
    'num_iterations'        : 100000,
    'iters_between_eval'    : 50,
    'batch_size'            : 100,
    'n_perms'               : 3,
    'n_pixels'              : 784,
    'layer_dims'            : [784,120,121,122,123,10],
    'test_reps'             : 50,
    'constant_b'            : True,

    # Omega parameters
    'xi'                    : 0.001,
    'omega_cost'            : 1.,

    # Dropout
    'keep_prob'             : 1
    }


def make_external_placeholders():
    feed = [[par['layer_dims'][n+1], par['layer_dims'][n], par['n_dendrites']] for n in range(par['n_hidden_layers'])]
    feed.append([par['layer_dims'][-1], par['layer_dims'][-2]])

    plc_weights = [[tf.placeholder_with_default(np.zeros(s, dtype=np.float32), shape=s) for i in range(par['n_perms'])] for s in feed]
    plc_omegas  = [[tf.placeholder_with_default(np.zeros(s, dtype=np.float32), shape=s) for i in range(par['n_perms'])] for s in feed]

    return plc_weights, plc_omegas


def update_dependencies():
    """
    Updates all parameter dependencies
    """
    par['num_layers'] = len(par['layer_dims'])
    par['n_hidden_layers'] = par['num_layers'] - 2

update_dependencies()

print("--> Parameters successfully loaded.\n")
