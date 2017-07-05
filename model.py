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

    def __init__(self, input_data, target_data, keep_prob):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = input_data
        self.target_data = target_data
        self.keep_prob = keep_prob # used for dropout

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    def run_model(self):

        self.x = self.input_data
        for n in range(par['num_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):
                W = tf.get_variable('W', (par['layer_dims'][n+1], par['layer_dims'][n]), initializer=tf.random_normal_initializer(0, par['init_weight_sd']))
                b = tf.get_variable('b', (par['layer_dims'][n+1], 1), initializer=tf.constant_initializer(0))
                if n == par['num_layers']-2:
                    # use linear activation function for last layer
                    self.y_hat = tf.matmul(W, self.x) + b
                else:
                    # use sigmoid activation function for all other layers
                    self.x = tf.nn.relu(tf.matmul(W, self.x) + b)
                if n == par['num_layers']-3:
                    # apply dropout right before final layer
                    self.x = tf.nn.dropout(self.x, self.keep_prob)


    def optimize(self):

        self.loss = tf.reduce_mean(tf.square(self.target_data - self.y_hat))
        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.minimize = opt.minimize(self.loss)

def main():

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    train_data = generate_data.Data(par['data_dir'] + par['data_filenames'][0])
    test_data = generate_data.Data(par['data_dir'] + par['data_filenames'][1])

    x = tf.placeholder(tf.float32, shape=[par['layer_dims'][0], par['batch_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[par['layer_dims'][-1], par['batch_size']]) # target data
    keep_prob = tf.placeholder(tf.float32) # used for dropout

    with tf.Session() as sess:

        model = Model(x, y, keep_prob)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        # keep track of the model performance across training
        train_performance = {'loss': [], 'trial': [], 'time': []}
        test_performance = {'loss': [], 'trial': [], 'time': []}

        for i in range(par['num_iterations']):

            # generate batch of N (batch_size X num_batches) trials
            input_data, target_data = train_data.generate_batch_data(test_data = False)

            if par['learning_rate']>0:

                """
                Train the model
                """
                _, train_loss, model_output = sess.run([model.minimize, model.loss, model.y_hat], {x: input_data.T, y: target_data.T, keep_prob: par['keep_prob']})

                """
                Test model on cross-validated data every 'iters_between_eval' trials
                """
                if (i+1)%par['iters_between_eval']==0:
                    test_loss = np.zeros((10))
                    for r in range(10):
                        test_input_data, test_target_data = train_data.generate_batch_data(test_data = True)
                        test_output = np.zeros((par['test_reps'], par['layer_dims'][-1], par['batch_size']), dtype = np.float32)
                        for j in range(par['test_reps']):
                            test_output[j,:,:] = sess.run(model.y_hat, {x: test_input_data[j,:,:].T, y: test_target_data.T, keep_prob: np.float32(1)})
                        test_output = np.mean(test_output, axis=0)
                        test_loss[r] = np.mean((test_output-test_target_data.T)**2)
                    test_performance = append_data(test_performance, np.mean(test_loss), time, i, t_start)
            else:
                loss, model_output = sess.run([model.loss, model.y_hat], {x: input_data, y: target_data[:,0]})

            train_performance = append_data(train_performance, train_loss, time, i, t_start)

            # reduce learning rate if train loss below thresholds
            if train_loss<60:
                par['learning_rate'] = 2e-4

            if (i+1)%par['iters_between_eval']==0:

                print_results(train_performance, test_performance)


def print_results(train_performance, test_performance):

    print('Trial {:7d}'.format(train_performance['trial'][-1]) +
      ' | Time {:0.2f} s'.format(train_performance['time'][-1]) +
      ' | Train loss {:0.4f}'.format(np.mean(train_performance['loss'][-par['iters_between_eval']:])) +
      ' | Test loss {:0.4f}'.format(test_performance['loss'][-1]))


def append_data(d, loss, time, i, t_start):

    d['loss'].append(loss)
    d['trial'].append(i*par['batch_size'])
    d['time'].append(time.time()-t_start)

    return d
