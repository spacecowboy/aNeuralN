from kalderstam.neural.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from kalderstam.neural.activation_functions import linear
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.cox_training import train_cox, generate_timeslots
from kalderstam.util.numpyhelp import indexOf

logger = logging.getLogger('kalderstam.neural.cox_training')

def generate_timeslots2(T):
    '''Slower, and can't trust IndexOf in case two outputs have the same value.
    But logically this is what generate_timeslots does.'''
    timeslots = numpy.zeros(len(T), dtype = int)
    sorted_T = numpy.sort(T, axis = 0)
    for i in range(len(timeslots)):
        timeslots[i] = indexOf(T, sorted_T[i])[0]

    return timeslots

def test(net, filename, epochs, learning_rate):
    logger.info("Running test for: " + str(epochs) + ", rate: " + str(learning_rate))
    P, T = parse_file(filename, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)
    #P = P[:100,:]
    #T = T[:100, :]

    timeslots = generate_timeslots(T)
    timeslots2 = generate_timeslots2(T)
    #Just to make sure it's correct when testing
    for x, y in zip(timeslots, timeslots2):
        assert(x == y)

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
    timeslots_target = generate_timeslots(T)
    timeslots_network = generate_timeslots(outputs)

#    plt.figure()
#    plt.title('Scatter between index ordering, epochs:rate | ' + str(epochs) + ':' + str(learning_rate))
#    plt.xlabel('Target timeslots')
#    plt.ylabel('Network timeslots')
#    plt.scatter(timeslots_target, timeslots_network, c = 'g', marker = 's')
#    plt.plot(timeslots_target, timeslots_target, 'r-')

    return net

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    p = 4 #number of input covariates
    net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON.ann')
    #net = build_feedforward(p, 10, 1, output_function = linear())
    lineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_no_noise.txt'
    nonlineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_no_noise.txt'
    lineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_with_noise.txt'
    nonlineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_with_noise.txt'


    net = test(net, lineartarget_nn, 500, 10)

    P, T = parse_file(lineartarget_nn, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)

    outputs = net.sim(P)
    timeslots_target = generate_timeslots(T)
    timeslots_network = generate_timeslots(outputs)

    plt.figure()
    plt.title('Scatter between index ordering, epochs:rate | ')
    plt.xlabel('Target timeslots')
    plt.ylabel('Network timeslots')
    plt.scatter(timeslots_target, timeslots_network, c = 'g', marker = 's')
    plt.plot(timeslots_target, timeslots_target, 'r-')
    plt.show()
