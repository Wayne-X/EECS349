import math
from node import Node
import sys


def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    See Textbook for algorithm.
    Make sure to handle unknown values, some suggested approaches were
    given in lecture.
    ========================================================================================================
    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
	maximum depth to search to (depth = 0 indicates that this node should output a label)
    ========================================================================================================
    Output: The node representing the decision tree learned over the given data set
    ========================================================================================================

    '''
    # stop if we hit limit, have homogeneous set, or gain nothing from a split
    if depth == 0:
        n = Node()
        n.label = mode(data_set)
        return n
    homo = check_homogenous(data_set)
    if homo is not None:
        n = Node()
        n.label = homo
        return n

    (attr, split_val) = pick_best_attribute(data_set, attribute_metadata, numerical_splits_count)
    if attr == False:
        n = Node()
        n.label = mode(data_set)
        return n
    # create a node with corresponding children
    n = Node()
    n.decision_attribute = attr
    n.is_nominal = split_val is False
    n.splitting_value = split_val
    n.name = attribute_metadata[attr]['name']
    n.mode = mode(([x[attr]] for x in data_set))
    if n.is_nominal:
        n.children = {k: ID3(v, attribute_metadata, numerical_splits_count, depth - 1) for (k, v) in
                      split_on_nominal(data_set, attr).items()}
    else:
        numerical_splits_count[attr] -= 1
        n.children = [ID3(x, attribute_metadata, numerical_splits_count, depth - 1) for x in
                      split_on_numerical(data_set, attr, split_val)]
    return n


def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the attribute at index 0 is the same for the data_set, if so return output otherwise None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
    '''
    # Your code here
    index0 = [x[0] for x in data_set]
    return index0[0] if index0 and index0.count(index0[0]) == len(index0) else None
    pass


# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: best attribute, split value if numeric
    ========================================================================================================
    '''
    # determine attributes we can use
    possible_attrs = [(i, x['is_nominal']) for (i, (x, n)) in enumerate(zip(attribute_metadata, numerical_splits_count))
                      if x['is_nominal'] or n > 0][1:]
    # find best gain and return result in correct format
    # keep missing nominal as it's own category, missing numerical is handled as mode in the helper
    gains = [(i, gain_ratio_nominal(data_set, i)) if n else (i,) + gain_ratio_numeric(data_set, i, 100) for (i, n) in possible_attrs]
    best = max(gains, key=lambda x: x[1]) if len(gains) > 0 else (0, 0)
    if best[1] == 0:
        return (False, False)
    elif len(best) == 2:
        return (best[0], False)
    else:
        return (best[0], best[2])


# # ======== Test Cases =============================
# numerical_splits_count = [20,20]
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
# data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
# data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

def mode(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of index 0.
    ========================================================================================================
    Output: mode of index 0.
    ========================================================================================================
    '''
    # Your code here
    index0 = [x[0] for x in data_set]
    counts = {}
    for i in index0:
        counts[i] = counts.get(i, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# mode(data_set) == 1
# data_set = [[0],[1],[0],[0]]
# mode(data_set) == 0

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''
    index0 = [x[0] for x in data_set]
    counts = {}
    for i in index0:
        counts[i] = counts.get(i, 0) + 1
    # reduce + list comprehension to do summation
    return reduce(lambda x, y: x + y,
                  (-x * math.log(x, 2) for x in (float(x) / len(index0) for x in counts.values())), 0)


# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    # Your code here
    groups = split_on_nominal(data_set, attribute)
    counts = {x: len(y) for x, y in groups.items()}
    attrs = set([x[attribute] for x in data_set])
    gain = entropy(data_set) - reduce(lambda x, y: x + y,
                                      [entropy(groups[x]) * counts[x] / len(data_set) for x in attrs], 0)
    intrinsic = reduce(lambda x, y: x + y,
                       [-x * math.log(x, 2) if x > 0 else 0 for x in [float(x) / len(data_set) for x in counts.values()]], 0)
    return gain / intrinsic if intrinsic != 0 else 0


# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps=1):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    # Your code here
    thresholds = [x[attribute] for x in data_set[::steps]]
    split = [split_on_numerical(data_set, attribute, t) for t in thresholds]

    gains = [entropy(data_set) - reduce(lambda x, y: x + y,
                                        [entropy(x) * len(x) / (len(s[0]) + len(s[1])) for x in s], 0) for s in split]
    intrinsics = [reduce(lambda x, y: x + y, [-x * math.log(x, 2) if x > 0 else 0 for x in
                                              [float(len(x)) / (len(s[0]) + len(s[1])) for x in s]], 0) for s in split]
    ratios = [g / i if i != 0 else 0 for (g, i) in zip(gains, intrinsics)]
    return max(zip(ratios, thresholds), key=lambda x: x[0])


# ======== Test case =============================
# data_set,attr,step = [[1,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.21744375685031775, 0.64)
# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)

def split_on_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, the index for a nominal attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    # Your code here
    groups = {}
    for i in data_set:
        if i[attribute] in groups:
            groups[i[attribute]].append(i)
        else:
            groups[i[attribute]] = [i]
    return groups


# ======== Test case =============================
# data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
# split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]}
# data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
# split on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]}

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, threshold (splitting) value
    ========================================================================================================
    Job:    Splits data_set into a tuple of two lists, the first list contains the examples where the given
	attribute has value less than the splitting value, the second list contains the other examples
    ========================================================================================================
    Output: Tuple of two lists as described above
    ========================================================================================================
    '''
    # missing treated as mode because it normally counts as -infinity for compares, and this seems less skewed
    mode_val = mode(([x[attribute]] for x in data_set if x is not None))
    return (filter(lambda x: (x[attribute] or mode_val) < splitting_value, data_set),
            filter(lambda x: (x[attribute] or mode_val) >= splitting_value, data_set))
    # ======== Test case =============================
    # d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
    # split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]])
    # d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
    # split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]])
