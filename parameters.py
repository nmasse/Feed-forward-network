import numpy as np
import os

print("--> Loading parameters...")

global par

"""
Independent parameters
"""
par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'debug_model'           : False,
    'load_previous_model'   : False,

    # Network configuration
    'layer_dimensions'      : [79,200,150,100,50,1], # Full is 'std_stf'
    'nonlinearity'          : 'sigmoid',       # Literature 0.8, for EI off 1

    }
