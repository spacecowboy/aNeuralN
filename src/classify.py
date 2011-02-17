import numpy
from kalderstam.util.filehandling import read_data_file, load_network,\
    parse_file
from kalderstam.neural.network import network, build_feedforward
from kalderstam.neural.matlab_functions import stat, plot2d2c
import logging
from kalderstam.neural.training_functions import traingd

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('classify')
    
    #data = read_data_file('P:\My Dropbox\Ann-Survival-Phd\Ecg1664_trn.dat')
    P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt", 5, ignorecols = [0,1,4], ignorerows = 0)
                
    net = build_feedforward(6, 3, 1)
    
    epochs = 100
    
    best = traingd(net, P, T, epochs)
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    #plot2d2c(best, P, T, 1)
    #plt.title("Only Gradient Descent.\n Total performance = " + str(total_performance) + "%")
    