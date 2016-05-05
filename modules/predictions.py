import copy
import os.path
from operator import xor
from parse import *


# DOCUMENTATION
# ========================================
# this function outputs predictions for a given data set.
# NOTE this function is provided only for reference.
# You will not be graded on the details of this function, so you can change the interface if 
# you choose, or not complete this function at all if you want to use a different method for
# generating predictions.

def create_predictions(tree, predict):
    '''
    Given a tree and a url to a data_set. Create a csv with a prediction for each result
    using the classify method in node class.
    '''
    predict_set, _ = parse(predict, True)
    for i in xrange(len(predict_set)):
        # classify
        predict_set[i][0] = tree.classify(predict_set[i])
        # return Missing attribute to ?
        for j in xrange(len(predict_set[i])):
            if predict_set[i][j] is None:
                predict_set[i][j] = "?"
    # NOTE: this saves the prediction as the first column. We manually move this to the last column using Excel
    return predict_set