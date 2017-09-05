"""
Nicolas Masse 2017

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

        # Load the input activity, the target data, and the training mask for
        # this batch of trials
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

        self.x = self.input_data
        for n in range(par['n_hidden_layers']):
            with tf.variable_scope('layer' + str(n)):
                print('\n-- Layer', n, '--')

                # Get layer variables
                W = tf.get_variable('W', (par['layer_dims'][n+1], par['layer_dims'][n], par['n_dendrites']), \
                    initializer=tf.random_normal_initializer(0, par['init_weight_sd']))
                b = tf.get_variable('b', (par['layer_dims'][n+1], 1), initializer=tf.constant_initializer(0)) if not par['constant_b'] else tf.constant(0.)

                # Run layer calculations
                x0 = tf.tensordot(W, self.x, ([1],[0]))
                #x1 = tf.nn.relu(x0 - self.dendrite_clamp)
                x1 = tf.nn.relu(x0)

                # Print layer variables and run final layer calculation
                self.x = tf.reduce_sum(x1,axis = 1) + b
                mu.tf_var_print(self.x, W, x0, x1)

                # Apply dropout right before final layer
                if n == par['n_hidden_layers']:
                    self.x = tf.nn.dropout(self.x, self.keep_prob)


        with tf.variable_scope('output'):
            print('\n-- Output --')

            # Get layer variables
            W = tf.get_variable('W', (par['layer_dims'][par['n_hidden_layers']+1], par['layer_dims'][par['n_hidden_layers']]), \
                initializer=tf.random_normal_initializer(0, par['init_weight_sd']))
            b = tf.get_variable('b', (par['layer_dims'][par['n_hidden_layers']+1], 1), initializer=tf.constant_initializer(0)) if not par['constant_b'] else tf.constant(0.)

            # Run layer calculation
            self.y = tf.matmul(W, self.x) + b
            mu.tf_var_print(W, b, self.y)


    def optimize(self):

        # Accumulate omega loss over all available weight matrices
        omega_loss = 0
        for layer, p in itertools.product(range(par['n_hidden_layers']+1), range(par['n_perms'])):
            sc = 'layer' + str(layer) if not layer == par['n_hidden_layers'] else 'output'
            with tf.variable_scope(sc, reuse = True):
                omega_loss += tf.reduce_sum(self.omegas[layer][p]*tf.square(self.ref_weights[layer][p] - tf.get_variable('W')))

        # Calculate loss and run optimization
        if par['optimizer'] == 'MSE':
            perf_loss = tf.reduce_mean(tf.square(self.target_data - self.y))
        elif par['optimizer'] == 'cross_entropy':
            perf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, labels = self.target_data, dim=0))

        # Aggregate total loss
        self.omega_loss = par['omega_cost']*omega_loss
        self.loss   = perf_loss + self.omega_loss

        # Create optimizer operation
        opt = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
        self.grads_and_vars = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(self.grads_and_vars)


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

        perm_ind    = 0
        prev_ind    = 0

        # Generate OmegaLayers and associated items
        omegas      = reg.create_omega_layer(np.arange(par['num_layers']-1))
        grad_list   = reg.init_gv_list(par['num_layers'])
        var_list    = reg.init_gv_list(par['num_layers'])
        w = []
        o = []

        task_switch = False

        for i in range(par['num_iterations']):

            # Generate batch of N (batch_size X num_batches) trials
            input_data, target_data = \
            stim.generate_batch_data(perm_ind=perm_ind, test_data=False)

            # Define the feed dict
            feed_dict = {**{x: input_data, y: target_data, keep_prob: par['keep_prob']}, \
                         **mu.zip_to_dict(plc_weights, list(w)),
                         **mu.zip_to_dict(plc_omegas, list(o))}

            # Train the model
            _, grads_and_vars, train_loss, model_output, omega_loss = \
                sess.run([model.train_op, model.grads_and_vars, model.loss, \
                          model.y, model.omega_loss], feed_dict)

            # Separate grads and vars for use in omega calculations
            grad_list, var_list = reg.sep_gv(grad_list, var_list, grads_and_vars)

            # Test model on cross-validated data every 'iters_between_eval' trials
            if i%par['iters_between_eval']==0 and i != 0:

                # Allocate test output data
                test_input  = np.zeros((par['n_perms'], par['n_pixels'], par['batch_size']), dtype=np.float32)
                test_target = np.zeros((par['n_perms'], par['layer_dims'][-1], par['batch_size']), dtype=np.float32)
                test_output = np.zeros((par['n_perms'], par['test_reps'], par['layer_dims'][-1], par['batch_size']), dtype=np.float32)

                # Loop over all available permutations and test the model on each
                for p in range(par['n_perms']):
                    test_input[p,:,:], test_target[p,:,:] = stim.generate_batch_data(perm_ind=p, test_data=True)
                    feed_dict = {x: test_input[p,:,:], y: test_target[p,:,:], keep_prob: np.float32(1)}
                    for j in range(par['test_reps']):
                        test_output[p,j,:,:] = sess.run(model.y, feed_dict)

                # Average over test repetitions
                test_output  = np.mean(test_output, axis=1)

                # Find accuracy and loss for each permutation
                acc_by_perm  = np.sum(np.float32(np.argmax(test_output, axis=1)==np.argmax(test_target, axis=1)), axis=1)/par['batch_size']
                loss_by_perm = np.mean((test_output - test_target)**2, axis=(1,2))

                # Print results for this test set
                mu.print_results(i, acc_by_perm, loss_by_perm, t_start, perm_ind)

            # Update the permutation index for the next iteration
            perm_ind = mu.update_pid(i)

            # If changing tasks, calculate omegas and reset accumulators
            if perm_ind != prev_ind:
                task_switch = True
                print('\nRunning omega calculation.')

                # This takes the grads and vars from the grad_list and var_list
                # format and rearranges them to be pairs of grads and vars as
                # addressed to each layer.  One gvs element will wind up being
                # a full grads_and_vars as needed by its appropriate OmegaLayer.
                gvs = reg.gen_gvs(grad_list, var_list)

                # The grad and var lists are reset in preparation for the next
                # sequence of updates
                grad_list   = reg.init_gv_list(par['num_layers'])
                var_list    = reg.init_gv_list(par['num_layers'])

                # Iterate over the OmegaLayers and the gvs list to process
                # each layer, get the appropriate reference weights and omega
                # values, and then change the active permutation
                w, o = reg.run_omegas_iteration(omegas, gvs, perm_ind)

                print('Omega calculation complete.\n')

            prev_ind = perm_ind


def calc_DC(o):
    """
    Keep whatever version that is desired:
    1) The first block of code returns indices of the least important dendrites for each neuron
    2) The second block of code returns a list of [neuron x dendrites]
       with only the least important dendrite set to 1, aka in a template format
    """
    # Version 1)
    DC = []
    for layer in range(par['n_hidden_layers']):
        DC.append(np.argmin(np.mean(np.max(np.stack(o[layer], axis=3), axis=3), axis=1), axis=1))
    return DC

    # Version 2)
    # DC = []
    # for layer in range(par['n_hidden_layers']):
    #     content = np.zeros([par['layer_dims'][layer+1]])


    #     for n, d in itertools.product(range(par['layer_dims'][layer+1]), range(par['n_dendrites'])):
    #         mean = np.mean(np.max(o[layer][:,n,:,d], axis=0)))

    #         ind = np.argmin(np.mean(o[layer][:,n,:,d], axis=1))
    #         content[n, d, ind] = 1
    #     DC.append(content)


try:
    main()
except KeyboardInterrupt:
    print('\nQuit by KeyboardInterrupt.')
