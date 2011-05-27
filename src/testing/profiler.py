from kalderstam.matlab.matlab_functions import loadsyn1, stat, plot2d2c, \
    loadsyn2, loadsyn3, plotroc, plot_network_weights
from kalderstam.util.filehandling import parse_file, save_network, load_network
from kalderstam.neural.network import build_feedforward, build_feedforward_committee
from random import uniform
import time
import numpy as np
from kalderstam.neural.training.cox_training import generate_timeslots, \
train_cox, plot_correctly_ordered
import matplotlib.pyplot as plt
from kalderstam.neural.activation_functions import linear
import logging
from kalderstam.util import graphlogger as glogger
from kalderstam.neural.error_functions.cox_error import get_risk_groups, \
    calc_sigma, total_error, get_beta_force, derivative, calc_beta

def test_cox_part(outputs, timeslots, epochs = 1, learning_rate = 2.0):
    np.seterr(all = 'raise') #I want errors!
    np.seterr(under = 'warn') #Except for underflows, just equate them to zero...
    risk_groups = get_risk_groups(timeslots)
    prev_error = None
    corrected = False
    der_value = np.zeros_like(outputs)
    for epoch in range(epochs):
        #logger.info("Epoch " + str(epoch))

        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError as e:
            print(str(e))
            break #Stop training

        sigma = calc_sigma(outputs)

        current_error = total_error(beta, sigma)

        if (prev_error is None or current_error <= prev_error):
            prev_error = current_error
            #Try increasing the rate, but less than below
            #learning_rate *= 1.2
            #logger.info('learning rate increased: ' + str(learning_rate))
        elif corrected:
            #Undo the weight correction
            outputs += learning_rate * der_value
            corrected = False
            #Half the learning rate
            learning_rate *= 0.5
            #if abs(learning_rate) < 0.1:
            #    learning_rate *= -10
            #And "redo" this epoch
            #logger.info('Halfing the learning rate: ' + str(learning_rate))
            continue

#        if corrected: #Only plot if the weight change was successful
        plot_correctly_ordered(outputs, timeslots)
        glogger.debugPlot('Total error', total_error(beta, sigma), style = 'b-')
        glogger.debugPlot('Sigma * Beta vs Epochs', beta * sigma, style = 'g-')
        glogger.debugPlot('Sigma vs Epochs', sigma, style = 'b-')
        glogger.debugPlot('Beta vs Epochs', beta, style = 'b-')
        #logger.info('Beta*Sigma = ' + str(sigma * beta))

        beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

        #Iterate over all output indices
        i = 0
        for output_index in [17, 77]:#range(len(outputs)):
            #logger.debug("Patient: " + str(i))
            i += 1

            der_value[output_index] = derivative(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)
        #"FIX" output
        outputs -= learning_rate * der_value
        print('value: ' + str(outputs[[77, 17], 0]))

        corrected = True

    #Just fix the last one as well
    try:
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
    except FloatingPointError as e:
        print(str(e))

    sigma = calc_sigma(outputs)

    current_error = total_error(beta, sigma)

#    if (prev_error is None or current_error <= prev_error):
#        prev_error = current_error
#    elif corrected:
#        #Undo the weight correction
#        outputs += learning_rate * der_value
#        corrected = False

    return outputs

def test():
    #numpy.seterr(all = 'raise')

    p = 4 #number of input covariates

    #net = build_feedforward(p, 8, 1, output_function = linear(1))
    net = load_network('/home/gibson/jonask/Projects/aNeuralN/ANNs/4x10x10x1.ann')

    filename = '/home/gibson/jonask/my_tweaked_fake_data_no_noise.txt'
    #filename = '/home/gibson/jonask/my_tweaked_fake_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_no_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_no_noise.txt'

    P, T = parse_file(filename, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)
    #P = P[:100,:]
    #T = T[:100, :]

    timeslots = generate_timeslots(T)

    outputs = net.sim(P)

    plot_network_weights(net)
    #plt.title('Before training, [hidden, output] vs [input, hidden, output\nError = ' + str(total_error(beta, sigma)))

    try:
        net = train_cox(net, (P, T), (None, None), timeslots, epochs = 10, learning_rate = 5)
    except FloatingPointError:
        print('Aaawww....')

    outputs = net.sim(P)


    plot_network_weights(net)

    plt.figure()
    plt.title('Scatter plot\n' + filename)
    plt.xlabel('Survival time years')
    plt.ylabel('Network output')
    try:
        plt.scatter(T.flatten(), outputs.flatten(), c = 'g', marker = 's')
    except:
        pass

#This is a test of the functionality in this file
if __name__ == '__main__':

    import pstats, cProfile

    #numpy.seterr(all = 'raise')
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    cProfile.runctx("test()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

    #test()
    #plt.show()
