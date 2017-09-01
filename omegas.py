"""
Synaptic Regularization Calculation
Source: Improved multitask learning through synaptic intelligence (Zenke et al)
Gregory Grant - Aug. 2017
"""

import numpy as np
from parameters import *
import itertools

"""
HOW TO USE

lid = layer id
pid = permutation id
process_iteration accumulates the grads and vars for that parameter matrix
    and calculates the full omega value
change_active_pid changes the permutation id to the indicated index
change_chron('all') releases the restriction that only previous permutations
    must be used in calculating omega, and is intended to be used when
    iterating over each permutation more that once

x = OmegaLayer(0)
capital_omega_for_perm_0 = x.process_iteration(grads_and_vars)
x.change_active_pid(1)
capital_omega_for_perm_1 = x.process_iteration(grads_and_vars)
...
capital_omega_for_perm_n = ...

x.change_chron('all')

x.change_active_pid(0)
capital_omega_for_perm_0 = x.process_iteration(grads_and_vars)
x.change_active_pid(1)
capital_omega_for_perm_1 = x.process_iteration(grads_and_vars)
...

"""

class OmegaObject:

    """
    One OmegaObject contains synaptic weight regularization about one network
    parameter set for one permutation.  It is responsible for calculating w_k,
    delta, and omega for that parameter set for that permutation.

    To calculate the total capital omega, all of the available OmegaObjects'
    omega values for that permutation are summed.
    """

    def __init__(self, size, lid, pid):
        self.lid    = lid               # Layer ID
        self.pid    = pid               # Permutation ID
        self.size   = size

        self.w_k    = np.zeros(size)    # Little omega value
        self.delta  = np.zeros(size)    # Denominator value
        self.omega  = np.zeros(size)    # Large omega value

        self.ref    = np.zeros(size)    # Ideal weight for this permutation
        self.grad   = np.zeros(size)    # Previous gradient (buffer)

    def add_to_w(self, grad, var):
        self.w_k   += np.multiply((self.ref-var), self.grad)
        self.grad   = grad
        self.ref    = var if np.sum(var) != 0 else self.ref

    def reset_w(self):
        self.w_k    = np.zeros(self.size)

    def calc_delta(self, prev_ref):
        self.delta  = self.ref - prev_ref

    def calc_omega(self):
        self.omega  = self.w_k/(self.delta**2 + par['xi'])


class OmegaLayer:

    """
    One OmegaLayer contains a set of OmegaObjects, one for each permutation for
    the indicated layer.  The OmegaLayer identifies an "active" permutation, and
    uses the active pid as its reference point for adding w_k values and summing
    the full omega.
    """

    def __init__(self, lid, active=0):
        self.lid        = lid
        self.active     = active
        self.full_omega = [0]*par['n_perms']
        self.chron      = 'prev_only'

        if lid != par['n_hidden_layers']:
            self.size = [par['layer_dims'][lid+1], par['layer_dims'][lid], par['n_dendrites']]
        else:
            self.size = [par['layer_dims'][lid+1], par['layer_dims'][lid]]

        self.omegas = []
        for p in range(par['n_perms']):
            self.omegas.append(OmegaObject(self.size, lid, p))

    def change_active_pid(self, pid):
        self.active = pid%par['n_perms']

    def change_chron(self, val):
        self.chron  = val               # Set to 'prev_only' or 'all'

    def reset_full_omega(self):
        self.full_omega = 0

    def get_perm(self, pid):
        return self.omegas[pid]

    def get_active_perm(self):
        return self.get_perm(self.active)

    def get_prev_perm_ref(self, pid):
        prev_pid = (pid-1)%par['n_perms']
        if self.chron == 'prev_only' and prev_pid > pid:
            return np.zeros(self.size)         # TODO : Default may not be zero
        else:
            return self.get_perm(prev_pid).ref

    def add_to_w(self, grad, var):
        active = self.get_active_perm()
        prev_ref = self.get_prev_perm_ref(self.active)

        active.add_to_w(grad, var)
        active.calc_delta(prev_ref)
        active.calc_omega()

    def reset_w(self):
        self.get_active_perm().reset_w()

    def calc_full_omega(self):
        """
        Calculate the capital omega for all tasks prior to the current active
        pid if the temporality is prev_only is true, or calculate for all tasks
        other than the active permutation.
        """
        for p, o in enumerate(self.omegas):
            self.full_omega[p] += o.omega

        return self.full_omega

    def process_iteration(self, grads_and_vars):
        # Note that these grads and vars are for a SINGLE parameter matrix
        # taken over the accumulation period, and are NOT the grads and vars
        # for the whole network graph

        # Apply all grads and vars to w_k (inc. last grad!)
        for gv in grads_and_vars:
            self.add_to_w(*gv)
        self.add_to_w(grads_and_vars[-1][0], 0.)     # This line is important!
                                                     # It ekes out the last w_k
                                                     # from the buffer in each
                                                     # OmegaObject instance

        self.full_omega = self.calc_full_omega()
        self.reset_w()

        return self.full_omega


@np.vectorize
def create_omega_layer(l):
    return OmegaLayer(l)
