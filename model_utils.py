import numpy as np
import tensorflow as tf
from parameters import *
import time

def update_pid(i):
    return (i//(2*par['iters_between_eval']))%par['n_perms']

def print_results(i, acc, loss, t_start, perm_ind):

    print('\n\nTrial {:8d}'.format(i*par['batch_size']) + ' | Time {:6.2f} s'.format(time.time() - t_start))
    print('\n   P | Acc.    | Loss')
    print('------------------------')
    for p in range(np.shape(acc)[0]):
        line = '{:4d} | '.format(p) + '{:0.4f}  | '.format(acc[p]) + '{:0.4f}'.format(loss[p])
        print(line, '<---' if p == perm_ind else '')
    print('')


def tf_var_print(*var):
    for v in var:
        print(str(v.name).ljust(20), v.shape)


def sum_of_list(l):
    acc = 0.
    for i in l:
        if type(i) == tuple or type(i) == list:
            acc += sum_of_list(i)
        else:
            acc += np.sum(i)
    return acc


def feed_dict_print(d):
    for name, item in d.items():
        print(str(name.name).ljust(30), str(name.shape).ljust(15), np.sum(item))


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


def print_elements(l):
    for i in l:
        if type(i) == list or type(i) == tuple:
            if type(i[0]) == list or type(i[0]) == tuple:
                print_elements(i)
            elif len(i) <= 79:
                print(i)
            else:
                s = split_list(i)
                print(str(s[0]) +'\n' + str(s[1]))
        else:
            if len(str(i)) <= 79:
                print(i)
            else:
                print(str(i[:len(str(i))//2]) + '\n' + str(i[len(str(i))//2:]))
