from kalderstam.neural.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from kalderstam.neural.activation_functions import linear
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta
from kalderstam.neural.training_functions import traingd_block
import kalderstam.util.graphlogger as glogger
import logging

logger = logging.getLogger('kalderstam.neural.cox_training')

def generate_timeslots(P, T):
    timeslots = numpy.array([], dtype = int)
    for x_index in range(len(P)):
        x = P[x_index]
        time = T[x_index][0]
        if len(timeslots) == 0:
            timeslots = numpy.insert(timeslots, 0, x_index)
        else:
            added = False
            #Find slot
            for time_index in timeslots:
                if time < T[time_index][0]:
                    timeslots = numpy.insert(timeslots, time_index, x_index)
                    added = True
                    break
            if not added:
                #Reached the end, insert here
                timeslots = numpy.append(timeslots, x_index)

    return timeslots

def test(net, filename, epochs):
    P, T = parse_file(filename, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)
    #P = P[:100,:]
    #T = T[:100, :]

    #timeslots = generate_timeslots(P, T)

    #outputs = net.sim(P)

    #beta, risk_outputs, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots)
    #sigma = calc_sigma(outputs)

    try:
        #net = train_cox(net, (P, T), (None, None), timeslots, epochs = 500, learning_rate = 5)
        net = traingd_block(net, (P, T), (None, None), epochs = epochs, learning_rate = 0.1, block_size = 0)
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
    except:
        pass

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    p = 4 #number of input covariates
    net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON.ann')
    #net = build_feedforward(p, 8, 1, output_function = linear(1))
    lineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_no_noise.txt'
    nonlineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_no_noise.txt'
    lineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_with_noise.txt'
    nonlineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_with_noise.txt'

    while True:
        test(net, lineartarget_nn, 1)
        plt.show()
