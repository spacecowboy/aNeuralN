from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta, generate_timeslots, \
    derivative, total_error, cox_pre_func, cox_block_func, cox_epoch_func, orderscatter, get_C_index
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.training.gradientdescent import traingd

logger = logging.getLogger('kalderstam.neural.cox_training')

def test(net, P, T, filename, epochs, learning_rate, block_size):
    logger.info("Running test for: " + filename + ' ' + str(epochs) + ", rate: " + str(learning_rate) + ", block_size: " + str(block_size))
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index = " + str(c_index))

    timeslots = generate_timeslots(T)

    try:
        #net = train_cox(net, (P, T), (None, None), timeslots, epochs, learning_rate = learning_rate)
        net = traingd(net, (P, T), (None, None), epochs, learning_rate, block_size, error_derivative = derivative, error_function
 = total_error, pre_loop_func = cox_pre_func, block_func = cox_block_func, epoch_func = cox_epoch_func)
    except FloatingPointError:
        print('Aaawww....')
    outputs = net.sim(P)
    c_index = get_C_index(T, outputs)
    logger.info("C index = " + str(c_index))

    plot_network_weights(net)

    plt.figure()
    plt.title('Scatter plot cox error\n' + filename + "\nC index = " + str(c_index))
    plt.xlabel('Survival time years')
    plt.ylabel('Network output')
    try:
        for t, o, e in zip(T[:, 0], outputs[:, 0], T[:, 1]):
            c = 'r'
            if e == 1:
                c = 'g'
            plt.scatter(t, o, c = c, marker = 's')
        #plt.scatter(T[:, 0].flatten(), outputs[:, 0].flatten(), c = 'g', marker = 's')
        plt.plot(T[:, 0].flatten(), T[:, 0].flatten(), 'r-')
    except:
        pass

    return net
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    filename = "/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt"
    P, T = parse_file(filename, targetcols = [4, 5], inputcols = [-1, -2, -3, -4], ignorerows = [0], normalize = True)
    #P = P[100:, :]
    #T = T[100:, :]

    p = len(P[0]) #number of input covariates
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_ALPHA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_OMEGA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_SIGMOID.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_FIXED.ann')

    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/4x10x10x1.ann')
    net = build_feedforward(p, 8, 1, output_function = 'linear')

    #Initial state
    outputs = net.sim(P)
    orderscatter(outputs, T, filename)
    plt.show()

    epochs = 20000
    rate = 5
    block_size = 100

    for times in range(100):
        net = test(net, P, T, filename, epochs, rate, block_size)

        outputs = net.sim(P)
        orderscatter(outputs, T, filename)
        glogger.setup()
        plt.show()
