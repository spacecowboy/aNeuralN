from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from kalderstam.neural.activation_functions import linear
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.training.cox_training import train_cox, generate_timeslots
from kalderstam.util.numpyhelp import indexOf

logger = logging.getLogger('kalderstam.neural.cox_training')

def generate_timeslots2(T):
    '''Slower, and can't trust IndexOf in case two outputs have the same value.
    But logically this is what generate_timeslots does.'''
    timeslots = numpy.zeros(len(T), dtype = int)
    sorted_T = numpy.sort(T, axis = 0)
    for i in range(len(timeslots)):
        timeslots[i] = indexOf(T, sorted_T[i])[0]

    return timeslots[::-1]

def experiment(net, P, T, filename, epochs, learning_rate):
    logger.info("Running experiment for: " + filename + ' ' + str(epochs) + ", rate: " + str(learning_rate))

    timeslots = generate_timeslots(T)

    try:
        net = train_cox(net, (P, T), (None, None), timeslots, epochs, learning_rate = learning_rate)
    except FloatingPointError:
        print('Aaawww....')
    outputs = net.sim(P)

    plot_network_weights(net)

    plt.figure()
    plt.title('Scatter plot cox error\n' + filename)
    plt.xlabel('Survival time years')
    plt.ylabel('Network output')
    try:
        plt.scatter(T.flatten(), outputs.flatten(), c = 'g', marker = 's')
        plt.plot(T.flatten(), T.flatten(), 'r-')
    except:
        pass
    #Manual test
    outputs = net.sim(P)

    return net

def orderscatter(net, T, filename):
    outputs = net.sim(P)
    timeslots_target = generate_timeslots(T)
    timeslots_network = generate_timeslots(outputs)
    network_timeslot_indices = []
    for output_index in timeslots_network:
        timeslot_index = indexOf(timeslots_target, output_index)
        network_timeslot_indices.append(timeslot_index)

    plt.figure()
    plt.title('Scatter between index ordering, initial\n' + str(filename))
    plt.xlabel('Target timeslots')
    plt.ylabel('Network timeslots')
    plt.plot(timeslots_target, timeslots_target, 'r-')
    plt.scatter(range(len(timeslots_target)), network_timeslot_indices, c = 'g', marker = 's')

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    p = 4 #number of input covariates
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_ALPHA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_OMEGA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_SIGMOID.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_FIXED.ann')
    net = build_feedforward(p, 20, 1, output_function = linear())
    lineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_no_noise.txt'
    nonlineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_no_noise.txt'
    lineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_with_noise.txt'
    nonlineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_with_noise.txt'
    productfunction_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/productfunction_no_noise.txt'
    productfunction_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/productfunction_with_noise.txt'

    #The training sample
    no_noise = productfunction_nn
    with_noise = productfunction_wn

    P, T_nn = parse_file(no_noise, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)
    P, T_wn = parse_file(with_noise, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)

    #Training sample
    T = T_wn
    filename = with_noise

    #Initial state
    orderscatter(net, T_nn, no_noise)
    plt.show()

    epochs = 10
    rate = 5

    for times in range(100):
        net = experiment(net, P, T, filename, epochs, rate)

        orderscatter(net, T_nn, no_noise)
        orderscatter(net, T_wn, with_noise)
        glogger.setup()
        plt.show()
