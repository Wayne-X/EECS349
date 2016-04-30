from random import shuffle
from ID3 import *
from operator import xor
from parse import parse
import matplotlib.pyplot as plt
import numpy as np
import os.path
from pruning import *


# NOTE: these functions are just for your reference, you will NOT be graded on their output
# so you can feel free to implement them as you choose, or not implement them at all if you want
# to use an entirely different method for graphing

def get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pct, prune):
    '''
    get_graph_accuracy_partial - Given a training set, attribute metadata, validation set, numerical splits count, and percentage,
    this function will return the validation accuracy of a specified (percentage) portion of the trainging setself.
    '''
    import random, copy
    length = len(train_set)
    if pct < 1 / float(length):
        return 0
    if pct >= (length - 1) / float(length):
        samples = (train_set,)
        iterations = 1
    else:
        samples = (random.sample(train_set, int(len(train_set) * pct)) for _ in xrange(iterations))
    if prune:
        total = sum([validation_accuracy(
            reduced_error_pruning(ID3(s, attribute_metadata, copy.copy(numerical_splits_count), depth), validate_set), validate_set)
                     for s in samples], 0)
    else:
        total = sum([validation_accuracy(ID3(s, attribute_metadata, copy.copy(numerical_splits_count), depth), validate_set)
                     for s in samples], 0)
    return total / iterations


def get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pcts, prune):
    '''
    Given a training set, attribute metadata, validation set, numerical splits count, iterations, and percentages,
    this function will return an array of the averaged graph accuracy partials based off the number of iterations.
    '''
    return [get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pct, prune)
            for pct in pcts]


# get_graph will plot the points of the results from get_graph_data and return a graph
def get_graph(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, lower, upper, increment):
    '''
    get_graph - Given a training set, attribute metadata, validation set, numerical splits count, depth, iterations, lower(range),
    upper(range), and increment, this function will graph the results from get_graph_data in reference to the drange
    percentages of the data.
    '''
    pcts = np.linspace(lower, upper, int((upper - lower) / increment) + 1).tolist()
    pruned = get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pcts, True)
    unpruned = get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pcts, False)

    plt.plot(pcts, pruned, 'k-x', label='pruned learning')
    plt.plot(pcts, unpruned, 'r-o', label='unpruned learning')
    plt.xlabel('Proportion of Training Data Used')
    plt.ylabel('Validation Accuracy')
    plt.title('Decision Tree Learning Curve')
    plt.legend(loc=4)
    plt.savefig('output/curve.png')
