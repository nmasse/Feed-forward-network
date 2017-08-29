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

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Reset TensorFlow before running anything
tf.reset_default_graph()

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, dendrite_clamp, keep_prob):

        print('\nBuilding graph...')

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = input_data
        self.target_data = target_data
        self.dendrite_clamp = dendrite_clamp
        self.keep_prob = keep_prob # used for dropout

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
                #b = tf.get_variable('b', (par['layer_dims'][n+1], 1), initializer=tf.constant_initializer(0))
                b = tf.constant(0.)

                # Run layer calculations
                x0 = tf.tensordot(W, self.x, ([1],[0]))
                #x1 = tf.nn.relu(x0 - self.dendrite_clamp)
                x1 = tf.nn.relu(x0)

                # Print layer variables and run final layer calculation
                self.x = tf.reduce_sum(x1,axis = 1) + b
                tf_var_print(self.x, W, x0, x1)

                # Apply dropout right before final layer
                if n == par['n_hidden_layers']:
                    self.x = tf.nn.dropout(self.x, self.keep_prob)


        with tf.variable_scope('output'):
            print('\n-- Output --')

            # Get layer variables
            W = tf.get_variable('W', (par['layer_dims'][par['n_hidden_layers']+1], par['layer_dims'][par['n_hidden_layers']]), \
                initializer=tf.random_normal_initializer(0, par['init_weight_sd']))
            #b = tf.get_variable('b', (par['layer_dims'][par['n_hidden_layers']+1], 1), initializer=tf.constant_initializer(0))
            b = tf.constant(0.)


            # Run layer calculation
            self.y = tf.matmul(W, self.x) + b
            tf_var_print(W, b, self.y)


    def optimize(self):

        # Calculate loss and run optimization
        self.loss = tf.reduce_mean(tf.square(self.target_data - self.y))
        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.grads_and_vars = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(self.grads_and_vars)


def main():

    # Create the stimulus class, and generate trial paramaters and input activity
    stim = generate_data.Data()

    x = tf.placeholder(tf.float32, shape=[par['layer_dims'][0], par['batch_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[par['layer_dims'][-1], par['batch_size']]) # target data
    dendrite_clamp = []
    keep_prob = tf.placeholder(tf.float32) # used for dropout

    # Open TensorFlow session
    with tf.Session() as sess:

        # Generate graph
        model = Model(x, y, dendrite_clamp, keep_prob)
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

        # Generate OmegaLayers
        omegas = reg.create_omega_layer(np.arange(par['num_layers']-1))

        for i in range(par['num_iterations']):

            # Generate batch of N (batch_size X num_batches) trials
            input_data, target_data = stim.generate_batch_data(perm_ind=perm_ind, test_data=False)

            # Train the model
            _, grads_and_vars, train_loss, model_output = sess.run([model.train_op, model.grads_and_vars, model.loss, model.y], \
                {x: input_data, y: target_data, keep_prob: par['keep_prob']})

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
                print_results(i, acc_by_perm, loss_by_perm, t_start, perm_ind)

            # Update the permutation index for the next iteration
            perm_ind = (i//(2*par['iters_between_eval']))%par['n_perms']

            # If changing tasks, calculate omegas and reset accumulators
            if perm_ind != prev_ind:

                pass    # Run omega calculation here

            prev_ind = perm_ind


def print_results(i, acc, loss, t_start, perm_ind):

    print('\n\nTrial {:8d}'.format(i*par['batch_size']) + ' | Time {:6.2f} s'.format(time.time() - t_start))
    print('\n   P | Acc.    | Loss')
    print('------------------------')
    for p in range(np.shape(acc)[0]):
        line = '{:4d} | '.format(p) + '{:0.4f}  | '.format(acc[p]) + '{:0.4f}'.format(loss[p])
        print(line, '<---' if p == perm_ind else '')


def tf_var_print(*var):
    for v in var:
        print(str(v.name).ljust(20), v.shape)


try:
    main()
except KeyboardInterrupt:
    print('Quit by KeyboardInterrupt.')
