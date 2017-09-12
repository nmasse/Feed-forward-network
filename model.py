"""
Nicolas Masse, Gregory Grant 2017
"""

import tensorflow as tf
import numpy as np
import generate_data
import time
from parameters import *
import os
import itertools
import omegas as reg
import model_utils as mu

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Reset TensorFlow before running anything
tf.reset_default_graph()

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, dendrite_clamp, keep_prob, plc_weights, plc_omegas):

        print('\nBuilding graph...')

        # Load the input activity, target data, etc. for this batch of trials
        self.input_data             = input_data
        self.target_data            = target_data
        self.dendrite_clamp         = dendrite_clamp
        self.keep_prob              = keep_prob             # used for dropout
        self.ref_weights            = plc_weights
        self.omegas                 = plc_omegas

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

        print('\nGraph built successfully.\n')


    def run_model(self):

        # Establish current tracked state, starting with the input activity
        self.x = self.input_data

        # Re-bind the chosen initialization functions
        w_init = tf.random_normal_initializer(0, par['init_weight_sd'])
        b_init = tf.constant_initializer(0)

        # Iterate through the model's hidden layers
        for n in range(par['n_hidden_layers']):
            with tf.variable_scope('layer' + str(n)):
                print('\n-- Layer', n, '--')

                # Establish the layer's shape
                w_shape = [par['layer_dims'][n+1], par['layer_dims'][n], par['n_dendrites']]
                b_shape = [par['layer_dims'][n+1], 1]

                # Get layer variables
                W = tf.get_variable('W', w_shape, initializer=w_init)
                b = tf.get_variable('b', b_shape, initializer=b_init)
                if par['constant_b']:
                    b = tf.constant(0.)

                # Run layer calculations
                x0      = tf.tensordot(W, self.x ([1],[0]))
                x1      = tf.reduce_sum(x0, axis=1) + b
                self.x  = tf.nn.relu(x1)

                # Print layer variables
                mu.tf_var_print(self.x, W, x0, x1)

        # Run the output layer
        with tf.variable_scope('output'):
            print('\n-- Output --')

            # Establish layer shape
            w_shape = [par['layer_dims'][par['num_layers']-1], par['layer_dims'][par['num_layers']-2]]
            b_shape = [par['layer_dims'][n+1], 1]

            # Get layer variables
            W = tf.get_variable('W', w_shape, initializer=w_init)
            b = tf.get_variable('b', b_shape, initializer=b_init)
            if par['constant_b']:
                b = tf.constant(0.)

            # Run layer calculations
            y0      = tf.matmul(W, self.x) + b
            self.y  = tf.nn.relu(y0)

            # Print layer variables
            mu.tf_var_print(self.y, W, b, y0)


    def optimize(self):

        # Accumulate omega loss
        self.omega_loss = tf.constant(0.)
        for layer, p in itertools.product(range(par['num_layers']-1), range(par['n_perms'])):
            sc = 'layer' + str(layer) if not layer == par['n_hidden_layers'] else 'output'
            with tf.variable_Scope(sc, reuse=True):
                sq = tf.square(self.ref_weights[layer][p] - tf.get_variable('W'))
                om = tf.multiply(self.omegas[layer][p], sq)
                self.omega_loss += tf.reduce_sum(om)

        # Calculate performance loss
        if par['optimizer'] == 'MSE':
            self.perf_loss = tf.square(self.target_data - self.y)
        elif par['optimizer'] == 'cross_entropy':
            setup = {'logits' : self.y, 'labels' : self.target_data, 'dim' : 0}
            self.perf_loss = tf.nn.softmax_cross_entropy_with_logits(**setup)

        # Aggregate total loss
        self.perf_loss  = tf.reduce_mean(self.perf_loss)
        self.omega_loss = tf.constant(par['omega_cost'])*self.omega_loss
        self.loss       = self.perf_loss + self.omega_loss

        # Create optimizer operation
        opt = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
        self.grads_and_vars = opt.compute_gradients(self.perf_loss)
        self.grads_and_vars_plus = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(self.grads_and_vars_plus)


def main():

    # Create the stimulus class, and generate trial paramaters and input activity
    stim = generate_data.Data()

    x = tf.placeholder(tf.float32, shape=[par['layer_dims'][0], par['batch_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[par['layer_dims'][-1], par['batch_size']]) # target data
    dendrite_clamp = []
    keep_prob = tf.placeholder(tf.float32) # used for dropout
    plc_weights, plc_omegas = make_external_placeholders()

    # Open TensorFlow session
    with tf.Session() as sess:

        # Generate graph
        model = Model(x, y, dendrite_clamp, keep_prob, plc_weights, plc_omegas)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        # Keep track of the model performance across training
        train_performance = {'loss': [], 'trial': [], 'time': []}
        test_performance = {'loss': [], 'trial': [], 'time': []}

        # Generate OmegaLayers, associated lists, and permutation indices
        interface   = reg.OmegaInterface(par['num_layers']-1)
        w           = []
        o           = []
        perm_ind    = 0
        prev_ind    = 0

        # Run model over its iterations
        for i in range(par['num_iterations']):

            # Generate batch of N (batch_size X num_batches) trials
            input_data, target_data = \
            stim.generate_batch_data(perm_ind=perm_ind, test_data=False)

            # Define the feed dict
            feed_dict = {**{x: input_data, y: target_data, keep_prob: par['keep_prob']}, \
                         **mu.zip_to_dict(plc_weights, w),
                         **mu.zip_to_dict(plc_omegas, o)}

            # Train the model
            _, grads_and_vars, train_loss, model_output, omega_loss = \
                sess.run([model.train_op, model.grads_and_vars, model.perf_loss, \
                          model.y, model.omega_loss], feed_dict)

            # Send the new grads and vars to the omegas for later use
            interface.accumulate_gvs(grads_and_vars)

            print(str(np.round(train_loss, 4)).ljust(8), str(np.round(omega_loss, 4)).ljust(8), end='\r')

            # Test model on cross-validated data every 'iters_between_eval' trials
            if i%par['iters_between_eval']==0 and i != 0:

                # Allocate test output data
                test_shape_in  = (par['n_perms'], par['test_reps'], par['layer_dims'][0], par['batch_size'])
                test_shape_out = (par['n_perms'], par['test_reps'], par['layer_dims'][-1], par['batch_size'])
                test_input  = np.zeros(test_shape_in, dtype=np.float32)
                test_target = np.zeros(test_shape_out, dtype=np.float32)
                test_output = np.zeros(test_shape_out, dtype=np.float32)

                # Loop over all available permutations and test the model on each
                for p, j in itertools.product(range(par['n_perms']), range(par['test_reps'])):
                    test_input[p,j,:,:], test_target[p,j,:,:] = stim.generate_batch_data(perm_ind=p, test_data=True)
                    feed_dict = {x: test_input[p,j,:,:], y: test_target[p,j,:,:], keep_prob: 1.}
                    test_output[p,j,:,:] = sess.run(model.y, feed_dict)

                    # Show model progress
                    progress = (p*par['test_reps']+j+1)/(par['n_perms']*par['test_reps'])
                    bar = int(np.round(progress*20))
                    print("Testing Model:\t [{}] ({:>3}%)".format("#"*bar + " "*(20-bar), int(np.round(100*progress))), end='\r')
                print("\nTesting session complete.\n")

                # Calculate accuracy and loss for this test set
                raw_comp     = np.float32(np.argmax(test_output, axis=2)==np.argmax(test_target, axis=2))
                acc_by_perm  = np.sum(raw_comp, axis=(1,2))/(par['batch_size']*par['test_reps'])
                loss_by_perm = np.mean((test_output - test_target)**2, axis=(1,2,3))

                # Print results for this test set
                mu.print_results(i, acc_by_perm, loss_by_perm, t_start, perm_ind)

            # Update the permutation index for the next iteration
            perm_ind = mu.update_pid(i)

            # If changing tasks, calculate omegas and reset accumulators
            if perm_ind != prev_ind:
                print('\nRunning omega calculation.')

                w, o = interface.run_iteration(perm_ind)

                print('Omega calculation complete.\n')

            prev_ind = perm_ind

try:
    main()
except KeyboardInterrupt:
    print('\nQuit by KeyboardInterrupt.')
