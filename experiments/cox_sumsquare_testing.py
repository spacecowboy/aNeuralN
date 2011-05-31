from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta
from kalderstam.neural.training.gradientdescent import traingd
from kalderstam.neural.training.genetic import train_evolutionary
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.util.numpyhelp import indexOf

logger = logging.getLogger('kalderstam.neural.cox_training')

def experiment(net, filename, epochs):
    P, T = parse_file(filename, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)
    #P = P[:100,:]
    #T = T[:100, :]

    try:
        #net = train_cox(net, (P, T), (None, None), timeslots, epochs = 500, learning_rate = 5)
        net = traingd(net, (P, T), (None, None), epochs = epochs, learning_rate = 0.01, block_size = 0)
        #net = train_evolutionary(net, (P, T), (None, None), epochs = epochs)
    except FloatingPointError:
        print('Aaawww....')
    outputs = net.sim(P)

    plot_network_weights(net)

    plt.figure()
    plt.title('Scatter plot sum square error\n' + filename)
    plt.xlabel('Survival time years')
    plt.ylabel('Network output')
    try:
        plt.scatter(T.flatten(), outputs.flatten(), c = 'g', marker = 's')
        plt.plot(T.flatten(), T.flatten(), 'r-')
    except:
        pass

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.info)

    p = 4 #number of input covariates
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON.ann')
    net = build_feedforward(p, 10, 1, output_function = "linear")
    lineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_no_noise.txt'
    nonlineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_no_noise.txt'
    lineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_with_noise.txt'
    nonlineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_with_noise.txt'

    while True:
        experiment(net, nonlineartarget_wn, 500)
        plt.show()
