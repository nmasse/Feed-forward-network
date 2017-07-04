"""
Nicolas Masse 2017

"""

import tensorflow as tf
import numpy as np
import generate_data
import time
from parameters import *

# Reset TensorFlow before running anythin
tf.reset_default_graph()

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = input_data
        self.target_data = target_data

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    def run_model(self):

        self.x = self.input_data
        for n in range(par['num_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):
                W = tf.get_variable('W', (par['layer_dims'][n+1], par['layer_dims'][n]), initializer=tf.random_normal_initializer())
                b = tf.get_variable('b', (par['layer_dims'][n+1], 1), initializer=tf.constant_initializer(0))
                if n == par['num_layers']-2:
                    # use linear activation function for last layer
                    self.x = tf.matmul(W, self.x) + b
                else:
                    # use sigmoid activation function for all other layers
                    self.x = tf.sigmoid(tf.matmul(W, self.x) + b)

    def optimize(self):

        self.loss = tf.reduce_mean(tf.square(self.target_data - self.x))
        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.minimize = opt.minimize(self.loss)

def main():

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    data = generate_data.Data(par['data_dir'] + par['data_filenames'][0])

    x = tf.placeholder(tf.float32, shape=[par['layer_dims'][0], par['batch_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[par['layer_dims'][-1], par['batch_size']]) # target data

    with tf.Session() as sess:

        model = Model(x, y)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        # keep track of the model performance across training
        model_performance = {'loss': [], 'trial': [], 'time': []}

        for i in range(par['num_iterations']):

            # generate batch of N (batch_size X num_batches) trials
            input_data, target_data = data.generate_batch_data()

            if par['learning_rate']>0:
                _, loss, model_output = sess.run([model.minimize, model.loss, model.x], {x: input_data.T, y: target_data.T})
            else:
                loss, model_output = sess.run([model.loss, model.x], {x: input_data, y: target_data[:,0]})

            model_performance['loss'].append(loss)
            model_performance['trial'].append(i*par['batch_size'])
            model_performance['time'].append(time.time()-t_start)
            if i%10==0:
                print_results(model_performance)

def print_results(model_performance):

    print('Trial {:7d}'.format(model_performance['trial'][-1]) +
      ' | Time {:0.2f} s'.format(model_performance['time'][-1]) +
      ' | Loss {:0.4f}'.format(model_performance['loss'][-1]))
