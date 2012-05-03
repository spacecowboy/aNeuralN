'''
Created on Nov 2, 2011

@author: jonask
'''
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import numpy as np
from survival.plotting import scatter

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

def scatter_all_inputs(P, headers = None):
    if headers is None:
        #Number of columns
        headers = range(len(P[0,:]))
    rows = len(P[0,:]) - 1
    cols = rows
    #print "length of data: " + str(rows)
    
    currentRow = 0
    currentCol = 0
    currentPlt = 1
    
    gs = gridspec.GridSpec(rows, cols)
    #ax = plt.subplot(gs[0, 0])
        
    while (currentRow < rows):
        if currentCol > currentRow:
            currentCol = 0
            currentRow += 1
            continue
        currentPlt = (currentRow -1) * cols + currentCol
        #print currentRow, currentCol, currentPlt
            
        # Turn off axis values
        ax = plt.subplot(gs[currentRow, currentCol])
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])

        scatter(P[:, currentCol], P[:, currentRow + 1], ax = ax, plotSlope = False)
        if currentCol == currentRow:
            ax.set_title(headers[currentCol])
        else:
            ax.set_title('')
        #ax.set_xlabel(headers[currentCol])
        if currentCol == 0:
            ax.set_ylabel(headers[currentRow+1])                
        
        #Finish with this
        currentCol += 1
    
if __name__ == '__main__':
    from kalderstam.util.filehandling import parse_file
    
    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/publication_data/Two_thirds_of_the_n4369_dataset_with_logs_lymf.txt"
    columns = ('age', 'log(1+lymfmet)', 'n_pos', 'tumsize', 'log(1+er_cyt)', 'log(1+pgr_cyt)', 'pgr_cyt_pos', 
               'er_cyt_pos', 'size_gt_20', 'er_cyt', 'pgr_cyt', 'time')
    
    #filename = "/home/gibson/jonask/Projects/DataMaker/hard_survival_test.txt"    
    #columns = ('X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6',  'X7', 'X8', 'X9')
    #targets = ['censtime', 'event']
    #columns = ('time', 'censtime', 'noisytime', 'censnoisytime')
    targets = []
    
    P, T = parse_file(filename, targetcols = targets, inputcols = columns, normalize = True, separator = '\t',
                      use_header = True)
    plt.figure()
    scatter_all_inputs(P, columns)
    
    plt.show()