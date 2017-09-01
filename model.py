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

    def __init__(self, input_data, target_data, dendrite_clamp, keep_prob, ref_weights, omegas_ph):

        print('\nBuilding graph...')

        # Load the input activity, the target data, and the training mask for
        # this batch of trials
        self.input_data             = input_data
        self.target_data            = target_data
        self.dendrite_clamp         = dendrite_clamp
        self.keep_prob              = keep_prob             # used for dropout
        self.ref_weights            = ref_weights
        self.omegas                 = omegas_ph


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
                tf_var_print(self.x, W, x0, x1)

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
            tf_var_print(W, b, self.y)


    def optimize(self):

        # Accumulate omega loss over all available weight matrices
        omega_loss = 0
        for layer, p in itertools.product(range(par['n_hidden_layers']+1), range(par['n_perms'])):
            sc = 'layer' + str(layer) if not layer == par['n_hidden_layers'] else 'output'
            with tf.variable_scope(sc, reuse = True):
                omega_loss += tf.reduce_sum(self.omegas[layer][p]*tf.square(self.ref_weights[layer][p] - tf.get_variable('W')))


        # Calculate loss and run optimization
        perf_loss   = tf.reduce_mean(tf.square(self.target_data - self.y))
        #perf_loss   = tf.reduce_mean(tf.square(self.target_data - self.y))
        perf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, labels = self.target_data, dim=0))

        self.omega_loss = par['omega_cost']*omega_loss
        self.loss   = perf_loss + self.omega_loss

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
    ref_weights, omegas_ph = make_external_placeholders()

    # Open TensorFlow session
    with tf.Session() as sess:

        # Generate graph
        model = Model(x, y, dendrite_clamp, keep_prob, ref_weights, omegas_ph)
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
        grad_list   = [[]]*par['num_layers']
        var_list    = [[]]*par['num_layers']


        task_switch = False

        for i in range(par['num_iterations']):

            # Generate batch of N (batch_size X num_batches) trials
            input_data, target_data = stim.generate_batch_data(perm_ind=perm_ind, test_data=False)

            # Train the model
            if task_switch:

                _, grads_and_vars, train_loss, model_output, omega_loss = sess.run([model.train_op, model.grads_and_vars, \
                    model.loss, model.y, model.omega_loss], {**{x: input_data, y: target_data, keep_prob: par['keep_prob']}, \
                    **zip_to_dict(ref_weights, list(w)),**zip_to_dict(omegas_ph, list(o))})
            else:
                _, grads_and_vars, train_loss, model_output, omega_loss = sess.run([model.train_op, model.grads_and_vars, \
                    model.loss, model.y, model.omega_loss], {x: input_data, y: target_data, keep_prob: par['keep_prob']})



            print(train_loss, omega_loss)

            # Separate grads and vars for use in omega calculations
            for k, (g, v) in enumerate(grads_and_vars):
                if par['constant_b']:
                    grad_list[k].append(g)
                    var_list[k].append(v)
                elif not par['constant_b'] and k%2 == 0:
                    grad_list[k//2].append(g)
                    var_list[k//2].append(v)
                else:
                    pass

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
                task_switch = True
                print('\nRunning omega calculation.')

                # This takes the grads and vars from the grad_list and var_list
                # format and rearranges them to be pairs of grads and vars as
                # addressed to each layer.  One gvs element will wind up being
                # a full grads_and_vars as needed by its appropriate OmegaLayer.
                gvs = []
                for l in range(par['num_layers']-1):
                    gl = [k[l] for k in grad_list]
                    vl = [k[l] for k in var_list]
                    gv = [[g,v] for g, v in zip(gl, vl)]
                    gvs.append(gv)

                # The grad and var lists are reset in preparation for the next
                # sequence of updates
                grad_list = [[]]*par['num_layers']
                var_list  = [[]]*par['num_layers']

                # Iterate over the OmegaLayers and the gvs list to process
                # each layer, get the appropriate reference weights and omega
                # values, and then change the active permutation
                w = []
                o = []
                for layer, gv in zip(omegas, gvs):
                    layer.process_iteration(gv)
                    w.append([layer.get_prev_perm_ref(m) for m in range(par['n_perms'])])
                    """
                    for m in range(par['n_perms']):
                        print('sum W ', ' LEN W ', len(w), np.sum(layer.get_prev_perm_ref(m)))
                        w.append(layer.get_prev_perm_ref(m))
                    """
                    o.append(layer.calc_full_omega())

                    layer.change_active_pid(perm_ind)
                ""
                print('Omega calculation complete.\n')


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


def split_list(l):
    return l[:len(l)//2], l[len(l)//2:]


def zip_to_dict(g, s):
    r = {}
    if len(g) == len(s):
        for i in range(len(g)):
            #r[g[i]] = s[i]
            for j in range(len(g[0])):
                r[g[i][j]] = s[i][j]
    else:
        #print("Lists in zip_to_dict not same size.", end='\r')
        pass

    return r


def list_aspect(l, f):
    r = []
    for i in l:
        if type(i) == list or type(i) == tuple:
            r.append(list_aspect(i, f))
        else:
            r.append(f(i))

    return r


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
