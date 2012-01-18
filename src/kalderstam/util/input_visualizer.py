'''
Created on Nov 2, 2011

@author: jonask
'''
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def plot_input(data, header = ""):
    plt.figure()
    plt.title("\nMean: " + str(np.mean(data)) + " Std: " + str(np.std(data)))
    n, bins, patches = plt.hist(data, 50, normed = False, facecolor = 'green', alpha = 0.75)
    # add a 'best fit' line
    y = mlab.normpdf(bins, np.mean(data), np.std(data))
    l = plt.plot(bins, y, 'r--', linewidth = 1)

    plt.xlabel(header + " total: " + str(len(data)))
    plt.ylabel('Number of data points')

def plot_all_inputs(P, headers = None):
    for var in xrange(len(P[0, :])):
        header = headers[var] if headers is not None else ""
        plot_input(P[:, var], header)
    plt.show()
