# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:05:14 2012

@author: jonask
"""

import numpy as np

def normalizeArray(array):
    '''Returns a new array, will not modify existing array.
    Normalization is simply subtracting the mean and dividing by the standard deviation (for non-binary arrays).'''
    inputs = np.copy(array)
    #First we must determine which columns have real values in them
    #Basically, if it isn't a binary value by comparing to 0 and 1
    for col in xrange(len(inputs[0])):
        real = False
        for value in inputs[:, col]:
            if value != 0 and value != 1:
                real = True
                break #No point in continuing now that we know they're real
        if real:
            #Subtract the mean and divide by the standard deviation
            inputs[:, col] = (inputs[:, col] - np.mean(inputs[:, col])) / np.std(inputs[:, col])

    return inputs
    
def normalizeArrayLike(test_array, norm_array):
    ''' Takes two arrays, the first is the test set you wish to have normalized as the second array is normalized.
    Normalization is simply subtracting the mean and dividing by the standard deviation (for non-binary arrays).
    
    So what this method does is for every column in array1, subtract by the mean of array2 and divide by the STD of
    array2. Mean that both arrays have been subjected to the same transformation.'''
    if test_array.shape[1] != norm_array.shape[1] or len(test_array.shape) != 2 or len(norm_array.shape) != 2:
        #Number of columns did not match
        raise ValueError('Number of columns did not match in the two arrays.')
    test_inputs = np.copy(test_array)
    #First we must determine which columns have real values in them
    #Basically, if it isn't a binary value by comparing to 0 and 1
    for col in xrange(norm_array.shape[1]):
        real = False
        for value in norm_array[:, col]:
            if value != 0 and value != 1:
                real = True
                break #No point in continuing now that we know they're real
        if real:
            #Subtract the mean and divide by the standard deviation of the other array
             test_inputs[:, col] = (test_inputs[:, col] - np.mean(norm_array[:, col])) / np.std(norm_array[:, col])

    return test_inputs