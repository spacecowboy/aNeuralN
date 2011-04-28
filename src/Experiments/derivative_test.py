from kalderstam.neural.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural.network import build_feedforward
import time
import numpy as np
import matplotlib.pyplot as plt
from kalderstam.neural.activation_functions import linear
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.cox_training import train_cox, generate_timeslots, \
    test_cox_part
from kalderstam.util.numpyhelp import indexOf

logger = logging.getLogger('kalderstam.neural.derivative_test')

def test(outputs, filename, epochs, learning_rate, P, T):
    logger.info("Running test for: " + str(epochs) + ", rate: " + str(learning_rate))

    timeslots = generate_timeslots(T)

    try:
        outputs = test_cox_part(outputs, timeslots, epochs, learning_rate)
    except FloatingPointError:
        print('Aaawww....')

    plt.figure()
    plt.title('Scatter plot cox error\n' + filename)
    plt.xlabel('Survival time years')
    plt.ylabel('Network output')
    try:
        plt.scatter(T.flatten(), outputs.flatten(), c = 'g', marker = 's')
        plt.plot(T.flatten(), T.flatten(), 'r-')
    except:
        pass

    return outputs

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    lineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_no_noise.txt'
    nonlineartarget_nn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_no_noise.txt'
    lineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/lineartarget_with_noise.txt'
    nonlineartarget_wn = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_data_set/nonlineartarget_with_noise.txt'

    epochs = 25
    rate = -40


    P, T = parse_file(lineartarget_nn, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)
    P = P[:100, :]
    T = T[:100, :]

    #Completely random outputs
    outputs_random = np.random.random((len(T), 1))

    #Only two wrong
    timeslots = generate_timeslots(T)
    outputs_two = np.zeros_like(outputs_random)
    prev_index = timeslots[0]
    for index in timeslots[1:]:
        outputs_two[index, 0] = outputs_two[prev_index, 0] - 0.1
        prev_index = index
    #Change value of one
    print(timeslots[0])
    print(timeslots[99])
    target_index = 17

    outputs_two[target_index] += 10

    print('value: ' + str(outputs_two[target_index]))

    outputs = outputs_two

    for i in [1, -1]: #Do it twice!
        rate *= i
        test(outputs, lineartarget_nn, epochs, rate, P, T)

        timeslots_target = generate_timeslots(T)
        timeslots_network = generate_timeslots(outputs)
        network_timeslot_indices = []
        for output_index in timeslots_network:
            timeslot_index = indexOf(timeslots_target, output_index)
            network_timeslot_indices.append(timeslot_index)

        print('value: ' + str(outputs_two[target_index]))
        plt.figure()
        plt.title('Scatter between index ordering, epochs:rate | ' + str(epochs) + ':' + str(rate))
        plt.xlabel('Target timeslots')
        plt.ylabel('Network timeslots')
        plt.plot(timeslots_target, timeslots_target, 'r-')
        plt.scatter(range(len(timeslots_target)), network_timeslot_indices, c = 'g', marker = 's')
        plt.show()
