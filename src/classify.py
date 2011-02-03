import numpy
from kalderstam.util.filehandling import read_data_file
from kalderstam.neural.network import network, build_feedforward
from kalderstam.neural.matlab_functions import stat
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    #logger = logging.getLogger('classify')
    
    #data = read_data_file('P:\My Dropbox\Ann-Survival-Phd\Ecg1664_trn.dat')
    data = read_data_file('/export/home/jonask/Dropbox/Ann-Survival-Phd/Ecg1664_trn.dat')
   
    inputs = numpy.array(data)
    
    targets = numpy.array(inputs[:, 39], dtype='float64') #target is 40th column
    ids = inputs[:, 40] #id is 41st column
    inputs = numpy.array(inputs[:, :39], dtype='float64') #first 39 columns are inputs
    
    net = build_feedforward(39, 1, 1)
    
    net.traingd(inputs, targets, 10, 0.1)
    
    Y = net.sim(inputs)
    
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, targets)
    