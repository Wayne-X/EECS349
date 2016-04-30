from node import Node
from ID3 import *
from operator import xor


# Note, these functions are provided for your reference.  You will not be graded on their behavior,
# so you can implement them as you choose or not implement them at all if you want to use a different
# architecture for pruning.

def reduced_error_pruning(root, validation_set):
    '''
    take the a node, training set, and validation set and returns the improved node.
    You can implement this as you choose, but the goal is to remove some nodes such that doing so improves validation accuracy.
    '''
    # can't prune if we have a leaf node or run out of validation data
    if root.label or not validation_set:
        return root

    # candidate leaf node, return it if accuracy improves
    prune_leaf = Node()
    prune_leaf.label = mode(validation_set)
    if validation_accuracy(prune_leaf, validation_set) >= validation_accuracy(root, validation_set):
        return prune_leaf

    # otherwise prune the children recursively
    if root.is_nominal:
        root.children = {k: reduced_error_pruning(v, filter(lambda x: x[root.decision_attribute] == k, validation_set)) for
                         (k, v) in root.children.items()}
    else:
        root.children = [reduced_error_pruning(root.children[0],
                                               filter(lambda x: (x[root.decision_attribute] or root.mode) < root.splitting_value,
                                                      validation_set)),
                         reduced_error_pruning(root.children[1],
                                               filter(lambda x: (x[root.decision_attribute] or root.mode) >= root.splitting_value,
                                                      validation_set))]
    return root


def validation_accuracy(tree, validation_set):
    '''
    takes a tree and a validation set and returns the accuracy of the set on the given tree
    '''
    # Your code here
    correct = sum((1 if tree.classify(x) == x[0] else 0 for x in validation_set), 0)
    total = len(validation_set)

    return float(correct) / total
