'''
Created on Nov 2, 2011

@author: jonask
'''
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def plot_input(data):
    plt.figure()
    plt.title("Mean: " + str(np.mean(data)) + " Std: " + str(np.std(data)))
    n, bins, patches = plt.hist(data, 50, normed = 1, facecolor = 'green', alpha = 0.75)
    # add a 'best fit' line
    y = mlab.normpdf(bins, np.mean(data), np.std(data))
    l = plt.plot(bins, y, 'r--', linewidth = 1)

def plot_all_inputs(P):
    for var in xrange(len(P[0, :])):
        plot_input(P[:, var])
    plt.show()
