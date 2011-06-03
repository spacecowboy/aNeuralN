from kalderstam.matlab.matlab_functions import plot_network_weights
from kalderstam.util.filehandling import parse_file, load_network, save_network
from kalderstam.neural.network import build_feedforward
import time
import numpy
import matplotlib.pyplot as plt
from kalderstam.neural.error_functions.cox_error import calc_sigma, calc_beta, generate_timeslots, \
    derivative, total_error, cox_pre_func, cox_block_func, cox_epoch_func, censor_rndtest, get_C_index, orderscatter
import kalderstam.util.graphlogger as glogger
import logging
from kalderstam.neural.training.gradientdescent import traingd

logger = logging.getLogger('kalderstam.neural.cox_training')

def experiment(net, P, T, filename, epochs, learning_rate):
    logger.info("Running experiment for: " + filename + ' ' + str(epochs) + ", rate: " + str(learning_rate))
    print("Number of patients with events: " + str(T[:, 1].sum()))
    print("Number of censored patients: " + str((1 - T[:, 1]).sum()))

    timeslots = generate_timeslots(T)

    try:
        net = traingd(net, (P, T), (None, None), epochs, learning_rate, block_size = 100, error_derivative = derivative, error_function
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
        plt.scatter(T[:, 0].flatten(), outputs[:, 0].flatten(), c = 'g', marker = 's')
        plt.plot(T.flatten(), T.flatten(), 'r-')
    except:
        pass

    return net

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    p = 4 #number of input covariates
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_ALPHA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_OMEGA.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_SIGMOID.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/PERCEPTRON_FIXED.ann')
    #net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/4x10x10x1.ann')
    net = build_feedforward(p, 8, 1, output_function = 'linear')
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

    #Amount to censor
    ratio = 0.50

    T_nn = censor_rndtest(T_nn, ratio)
    T_wn = censor_rndtest(T_wn, ratio)

    #Training sample
    T = T_wn
    filename = with_noise

    #Initial state
    outputs = net.sim(P)
    orderscatter(outputs, T_nn, no_noise)
    plt.show()

    epochs = 10
    rate = 5

    for times in range(100):
        net = experiment(net, P, T, filename, epochs, rate)
        outputs = net.sim(P)

        orderscatter(outputs, T_nn, no_noise)
        orderscatter(outputs, T_wn, with_noise)
        glogger.setup()
        plt.show()
