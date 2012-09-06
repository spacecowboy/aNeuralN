# -*- coding: utf-8 -*-
'''
This is the python code which should be equivalent to the faster code in the c module.
This should only be used by default before an actual install
'''
from numpy import exp
from math import tanh

class Node(object):
    def __init__(self, active="linear", random_range=1, weights=None):
        if weights is None:
            weights = {}
        self.weights = weights
        self.random_range = random_range

        self.activation_function = active
        if active == 'logsig':
            self.function = logsig
            self.derivative = logsig_derivative
        elif active == 'tanh':
            self.function = tanh
            self.derivative = tanh_derivative
        else:
            self.function = linear
            self.derivative = linear_derivative
            self.activation_function = 'linear'

    def _inputsum(self, inputs):
        sum = 0
        for prev, weight in self.weights.iteritems():
            if isinstance(prev, BiasNode):
                sum += weight
            elif isinstance(prev, Node):
                sum += weight * prev.output(inputs)
            elif isinstance(prev, int):
                # It's an input index
                sum += inputs[prev] * weight
            else:
                raise TypeError('The previous node was neither a node nor an index')

        return sum

    def output(self, inputs):
        return self.function(self._inputsum(inputs))

    def output_derivative(self, inputs):
        return self.derivative(self.output(inputs))

class BiasNode(Node):
    def __init__(self):
        super(BiasNode, self).__init__()

    def output(self, inputs):
        return 1

    def output_derivative(self, inputs):
        return 0

def linear(x):
    return x

def linear_derivative(y):
    return 1

def logsig(x):
    return 1 / (1 + exp(-x))

def logsig_derivative(y):
    return y * (1 - y)

# Tanh defined in math

def tanh_derivative(y):
    return (1 - y) * (1 + y)
