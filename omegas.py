import numpy as np
from parameters import *
import itertools

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

        self.grad   = np.zeros(size)    # Previous gradient
        self.ref    = np.zeros(size)    # Ideal weight for this permutation
        self.prev   = np.zeros(size)    # Previous weight reference for w_k

    def set_ref(self, var):
        self.ref    = var

    def add_to_w(self, var, grad):
        self.w_k   += np.multiply((self.prev-var), self.grad)
        self.grad   = grad
        self.prev   = var

    def reset_w(self):
        self.w_k    = np.zeros(size)

    def calc_delta(self, prev_ref):
        self.delta  = self.ref - prev_ref

    def calc_omega(self):
        self.omega  = self.w_k/(self.delta + par['xi'])


class OmegaLayer:

    """
    One OmegaLayer contains a set of OmegaObjects, one for each permutation for
    the indicated layer.  The OmegaLayer identifies an "active" permutation, and
    uses the active pid as its reference point for adding w_k values and summing
    the full omega.
    """

    def __init__(self, lid, active=0):
        self.lid    = lid
        self.size   = [par['layer_dims'][lid+1], par['layer_dims'][lid], par['n_dendrites']]
        self.active = active

        self.full_omega = 0
        self.perms      = np.zeros(self.size)
        self.chron      = 'prev_only'

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

    def get_prev_perm_ref(self, pid):
        prev_pid = (pid-1)%par['n_perms']
        if self.chron == 'prev_only' and prev_pid > pid:
            return 0
        else:
            return get_perm(prev_pid).ref

    def add_to_w(self, var, grad):
        active = get_perm(self.active)
        prev_ref = get_prev_perm_ref(self.active)

        active.add_to_w(var, grad)
        active.calc_delta(prev_ref)
        active.calc_omega()

    def calc_full_omega(self):
        """
        Calculate the capital omega for all tasks prior to the current active
        pid if the temporality is prev_only is true, or calculate for all tasks
        other than the active permutation.
        """
        for o in self.omegas:
            if o.pid < self.active:
                self.full_omega += o.omega
            elif o.pid == self.active:
                pass
            elif o.pid > self.active and self.chron=='all':
                self.full_omega += o.omega
            else:
                pass

        return self.full_omega


@np.vectorize
def create_omega_layer(l):
    return OmegaLayer(l)

x = OmegaLayer(0)
print(x.size, x.active)
x.change_active_pid(2)
print(x.size, x.active)









quit()
