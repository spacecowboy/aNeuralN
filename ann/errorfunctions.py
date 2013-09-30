"""Total error = E, and derivative equals ej, where:
    E = sum of all Ej
    Ej = 1/2 * (ej)^2
    ej = result(j) - target(j)."""
    
from __future__ import division

import numpy as np

class SSE():
    '''Sum Square error'''
    def __call__(self, target, result):
        return np.sum((target - result) ** 2) / 2

    def derivative(self, target, result):
        return np.sum(target - result)

def sumsquare_total(target, result):
    try:
        return ((result[:, 0] - target[:, 0]) ** 2).sum() / 2
    except FloatingPointError as e:
        print(target, result)
        raise e

def sumsquare_derivative(targets, results, index):
    """dE/dej = ej = result - target."""
    return (results[index, 0] - targets[index, 0])
