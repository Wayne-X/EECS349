# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 if numeric and a dictionary if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index holds examples >= the splitting value
#
# label - is None if there is a decision attribute, and is the output label (0 or 1 for
#	    the homework data set) if there are no other attributes
#       to split on or the data is homogenous
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.splitting_value = None
        self.children = {}
        self.name = None
        # added since we default to the mode for missing attributes
        self.mode = None

    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''
        # Your code here
        if self.label is not None:
            return self.label
        if self.is_nominal:
            # Use mode for unknown values
            return self.children.get(instance[self.decision_attribute], self.children.get(self.mode, None)).classify(instance)
        # treat missing as mode
        return self.children[0 if (instance[self.decision_attribute] or self.mode) < self.splitting_value else 1].classify(instance)

    def print_tree(self, indent=0):
        '''
        returns a string of the entire tree in human readable form
        '''
        # Your code here
        if self.label is not None:
            return " " * indent + str(self.label)
        if self.is_nominal:
            return "\n".join(
                [" " * indent + "{0} = {1}\n{2}".format(self.name, k, v.print_tree(indent + 2)) for (k, v) in self.children.items()])
        return "\n".join(
            [" " * indent + "{0} {1} {2}\n{3}".format(self.name, k, self.splitting_value, v.print_tree(indent + 2)) for (k, v) in
             zip(["<", ">="], self.children)])

    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        return " OR ".join(["({0})".format(x) for x in self.print_dnf_tree_helper("")])
        pass

    def print_dnf_tree_helper(self, context):
        '''
        returns the disjunct normalized form of the tree.
        '''
        if self.label is not None:
            return [context[5:]] if self.label == 1 else []

        if self.is_nominal:
            return sum([v.print_dnf_tree_helper("{0} AND {1} = {2}".format(context, self.name, k)) for (k, v) in self.children.items()], [])

        return sum([v.print_dnf_tree_helper("{0} AND {1} {2} {3}".format(context, self.name, k, self.splitting_value)) for (k, v) in
                    zip(["<", ">="], self.children)], [])
