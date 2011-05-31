from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta, generate_timeslots, \
    derivative, total_error, cox_pre_func, cox_block_func
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.training.cox_training import train_cox
from kalderstam.util.numpyhelp import indexOf
from kalderstam.neural.training.gradientdescent import traingd

logger = logging.getLogger('kalderstam.neural.cox_training')

def generate_timeslots2(T):
    '''Slower, and can't trust IndexOf in case two outputs have the same value.
    But logically this is what generate_timeslots does.'''
    timeslots = numpy.zeros(len(T), dtype = int)
    sorted_T = numpy.sort(T, axis = 0)
    for i in range(len(timeslots)):
        timeslots[i] = indexOf(T, sorted_T[i])[0]

    return timeslots[::-1]

def test(net, P, T, filename, epochs, learning_rate):
    logger.info("Running test for: " + filename + ' ' + str(epochs) + ", rate: " + str(learning_rate))

    timeslots = generate_timeslots(T)

    try:
        #net = train_cox(net, (P, T), (None, None), timeslots, epochs, learning_rate = learning_rate)
        net = traingd(net, (P, T), (None, None), epochs, learning_rate, block_size = 0, error_derivative = derivative, error_function = total_error, pre_loop_func = cox_pre_func, block_func = cox_block_func)
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

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"
    P, T = parse_file(filename, targetcols = [4], inputcols = [-1, -2, -3, -4], ignorerows = [0], normalize = True)

    #P = P[100:, :]
    #T = T[100:, :]

    p = len(P[0]) #number of input covariates
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_ALPHA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_OMEGA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_SIGMOID.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_FIXED.ann')

    net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/4x10x10x1.ann')
    #net = build_feedforward(p, 20, 1, output_function = linear())

    #Initial state
    orderscatter(net, T, filename)
    plt.show()

    epochs = 2000
    rate = 5

    for times in range(100):
        net = test(net, P, T, filename, epochs, rate)

        orderscatter(net, T, filename)
        glogger.setup()
        plt.show()
