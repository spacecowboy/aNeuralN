from numpy import log

def total_error(target, result):
    return -sum(target*log(result) + (1 - target)*log(1-result))

def derivative(target, result):
    '''This result is however only correct when combined with a logistic activation function on the output node.'''
    return (result - target)