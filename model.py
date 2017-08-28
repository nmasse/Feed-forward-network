"""
Nicolas Masse 2017

"""

import tensorflow as tf
import numpy as np
import generate_data
import time
from parameters import *
import os

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
                b = tf.get_variable('b', (par['layer_dims'][n+1], 1), initializer=tf.constant_initializer(0))

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
            b = tf.get_variable('b', (par['layer_dims'][par['n_hidden_layers']+1], 1), initializer=tf.constant_initializer(0))


            # Run layer calculation
            self.y = tf.matmul(W, self.x) + b
            tf_var_print(W, b, self.y)


    def optimize(self):

        # Calculate loss and run optimization
        self.loss = tf.reduce_mean(tf.square(self.target_data - self.y))
        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.minimize = opt.minimize(self.loss)


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

        for i in range(par['num_iterations']):

            # Generate batch of N (batch_size X num_batches) trials
            perm_ind = (i//250)%par['n_perms']
            input_data, target_data = stim.generate_batch_data(perm_ind=perm_ind, test_data=False)

            # Train the model
            _, train_loss, model_output = sess.run([model.minimize, model.loss, model.y], \
                {x: input_data, y: target_data, keep_prob: par['keep_prob']})

            # Append performance data
            train_performance = append_data(train_performance, train_loss, time, i, t_start)

            # Test model on cross-validated data every 'iters_between_eval' trials
            if (i+1)%par['iters_between_eval']==0:
                test_loss = np.zeros((10))
                for r in range(10):
                    # Generate batch of trials
                    test_input_data, test_target_data = stim.generate_batch_data(test_data=True)
                    test_output = np.zeros((par['test_reps'], par['layer_dims'][-1], par['batch_size']), dtype=np.float32)

                    # Test the model
                    for j in range(par['test_reps']):
                        test_output[j,:,:] = sess.run(model.y, {x: test_input_data[:,:], y: test_target_data, keep_prob: np.float32(1)})

                    # Average across test reps. and calculate MSE loss
                    test_output = np.mean(test_output, axis=0)
                    test_loss[r] = np.mean((test_output-test_target_data)**2)

                # Append performance data
                test_performance = append_data(test_performance, np.mean(test_loss), time, i, t_start)

            # Reduce learning rate if train loss below thresholds
            if train_loss<60:
                par['learning_rate'] = 2e-4

            # Print results and associated data
            if i%par['iters_between_eval']==0 and i != 0:
                print_results(train_performance, test_performance, perm_ind)


def print_results(train_performance, test_performance, perm_ind):

    print('Trial {:7d}'.format(train_performance['trial'][-1]) +
      ' | Perm. {:2d}'.format(perm_ind) +
      ' | Time {:6.2f} s'.format(train_performance['time'][-1]) +
      ' | Train loss {:0.4f}'.format(np.mean(train_performance['loss'][-par['iters_between_eval']:])) +
      ' | Test loss {:0.4f}'.format(test_performance['loss'][-1]))

def append_data(d, loss, time, i, t_start):

    d['loss'].append(loss)
    d['trial'].append(i*par['batch_size'])
    d['time'].append(time.time()-t_start)

    return d

def tf_var_print(*var):
    for v in var:
        print(str(v.name).ljust(20), v.shape)

try:
    main()
except KeyboardInterrupt:
    print('Quit by KeyboardInterrupt.')
