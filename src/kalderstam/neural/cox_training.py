from kalderstam.neural.error_functions.cox_error import derivative, calc_beta, \
    calc_sigma, get_risk_outputs, total_error, get_risk_groups, get_beta_force
from kalderstam.util.decorators import benchmark
import logging
from numpy import exp
import numpy as np
import kalderstam.util.graphlogger as glogger
from kalderstam.neural.training_functions import traingd_block
from kalderstam.util.filehandling import normalizeArray

logger = logging.getLogger('kalderstam.neural.cox_training')
prev_timeslots_network = None

def generate_timeslots(T):
    timeslots = np.array([], dtype = int)
    for x_index in range(len(T)):
        time = T[x_index][0]
        if len(timeslots) == 0:
            timeslots = np.insert(timeslots, 0, x_index)
        else:
            added = False
            #Find slot
            for index in range(len(timeslots)):
                time_index = timeslots[index]
                if time > T[time_index, 0]:
                    timeslots = np.insert(timeslots, index, x_index)
                    added = True
                    break
            if not added:
                #Reached the end, insert here
                timeslots = np.append(timeslots, x_index)

    return timeslots

def plot_correctly_ordered(outputs, timeslots):
    timeslots_network = generate_timeslots(outputs)
    global prev_timeslots_network
    if prev_timeslots_network == None:
        prev_timeslots_network = timeslots_network
    #Now count number of correctly ordered indices
    count = 0
    diff = 0
    for i, j, prev in zip(timeslots, timeslots_network, prev_timeslots_network):
        if i == j:
            count += 1
        if j != prev:
            diff += 1

    glogger.debugPlot('Network ordering difference', y = diff, style = 'r-')
    logger.info('Network ordering difference: ' + str(diff))
    prev_timeslots_network = timeslots_network

    countreversed = 0
    for i, j in zip(timeslots[::-1], timeslots_network):
        if i == j:
            countreversed += 1
    correct = max(count, countreversed)
    #glogger.debugPlot('Number of correctly ordered outputs', y = correct, style = 'r-')
    #logger.info('Number of correctly ordered outputs: ' + str(correct))

def train_cox(net, (test_inputs, test_targets), (validation_inputs, validation_targets), timeslots, epochs = 1, learning_rate = 2.0):
    np.seterr(all = 'raise') #I want errors!
    np.seterr(under = 'warn') #Except for underflows, just equate them to zero...
    inputs = test_inputs
    outputs = net.sim(inputs)
    risk_groups = get_risk_groups(timeslots)
    prev_error = None
    corrected = False
    for epoch in range(epochs):
        logger.info("Epoch " + str(epoch))
        outputs = net.sim(inputs)
        #Check if beta will diverge here, if so, end training with error 0
        #if beta_diverges(outputs, timeslots):
            #End training
        #    logger.info('Beta diverges...')
        #    break

        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError as e:
            print(str(e))
            break #Stop training

        sigma = calc_sigma(outputs)

        current_error = total_error(beta, sigma)

        if (prev_error == None or current_error <= prev_error):
            prev_error = current_error
        elif corrected:
            #Undo the weight correction
            apply_weight_corrections(net, -learning_rate)
            corrected = False
            #Half the learning rate
            learning_rate *= -0.5
            #And "redo" this epoch
            logger.info('Halfing the learning rate: ' + str(learning_rate))
            continue

        if corrected: #Only plot if the weight change was successful
            plot_correctly_ordered(outputs, timeslots)
            glogger.debugPlot('Total error', total_error(beta, sigma), style = 'b-')
            glogger.debugPlot('Sigma * Beta vs Epochs', beta * sigma, style = 'g-')
            glogger.debugPlot('Sigma vs Epochs', sigma, style = 'b-')
            glogger.debugPlot('Beta vs Epochs', beta, style = 'b-')
        logger.info('Beta*Sigma = ' + str(sigma * beta))

        beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

        #Set corrections to 0 on all nodes first
        for node in net.get_all_nodes():
            node.weight_corrections = {}

        #Iterate over all output indices
        i = 0
        for input, output_index in zip(inputs, range(len(outputs))):
            logger.debug("Patient: " + str(i))
            i += 1
            #Set error to 0 on all nodes first
            for node in net.get_all_nodes():
                node.error_gradient = 0

            #Set errors on output nodes first
            for node, gradient in zip(net.output_nodes, derivative(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)):
                #glogger.debugPlot('Gradient', gradient, style = 'b.')
                node.error_gradient = gradient

            #Iterate over the nodes and correct the weights
            for node in net.output_nodes + net.hidden_nodes:
                #Calculate local error gradient
                node.error_gradient *= node.output_derivative(input)

                #Propagate the error backwards and then update the weights
                for back_node, back_weight in node.weights.items():

                    if back_node not in node.weight_corrections:
                        node.weight_corrections[back_node] = []

                    try:
                        index = int(back_node)
                        node.weight_corrections[back_node].append(node.error_gradient * input[index])
                    except ValueError:
                        back_node.error_gradient += back_weight * node.error_gradient
                        node.weight_corrections[back_node].append(node.error_gradient * back_node.output(input))

                #Finally, correct the bias
                if "bias" not in node.weight_corrections:
                    node.weight_corrections["bias"] = []
                node.weight_corrections["bias"].append(node.error_gradient)

        apply_weight_corrections(net, learning_rate)
        corrected = True

    return net

def apply_weight_corrections(net, learning_rate):
    #Iterate over the nodes and correct the weights
    for node in net.output_nodes + net.hidden_nodes:
        #Calculate weight update
        for back_node, back_weight in node.weights.items():
            node.weights[back_node] = back_weight + learning_rate * sum(node.weight_corrections[back_node]) / len(node.weight_corrections[back_node])
        #Don't forget bias
        node.bias = node.bias + learning_rate * sum(node.weight_corrections["bias"]) / len(node.weight_corrections["bias"])

def test_cox_part(outputs, timeslots, epochs = 1, learning_rate = 2.0):
    np.seterr(all = 'raise') #I want errors!
    np.seterr(under = 'warn') #Except for underflows, just equate them to zero...
    risk_groups = get_risk_groups(timeslots)
    prev_error = None
    corrected = False
    der_value = np.zeros_like(outputs)
    for epoch in range(epochs):
        logger.info("Epoch " + str(epoch))

        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError as e:
            print(str(e))
            break #Stop training

        sigma = calc_sigma(outputs)

        current_error = total_error(beta, sigma)

        if (prev_error == None or current_error <= prev_error):
            prev_error = current_error
        elif corrected:
            #Undo the weight correction
            outputs -= learning_rate * der_value
            corrected = False
            #Half the learning rate
            learning_rate *= 0.5
            #if abs(learning_rate) < 0.01:
            #    learning_rate *= -1000
            #And "redo" this epoch
            logger.info('Halfing the learning rate: ' + str(learning_rate))
            continue

#        if corrected: #Only plot if the weight change was successful
        plot_correctly_ordered(outputs, timeslots)
        glogger.debugPlot('Total error', total_error(beta, sigma), style = 'b-')
        glogger.debugPlot('Sigma * Beta vs Epochs', beta * sigma, style = 'g-')
        glogger.debugPlot('Sigma vs Epochs', sigma, style = 'b-')
        glogger.debugPlot('Beta vs Epochs', beta, style = 'b-')
        logger.info('Beta*Sigma = ' + str(sigma * beta))

        beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

        #Iterate over all output indices
        i = 0
        for output_index in range(len(outputs)):
            logger.debug("Patient: " + str(i))
            i += 1

            der_value[output_index] = derivative(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)
        #"FIX" output
        outputs += learning_rate * der_value

        corrected = True

    return outputs

def test():
    from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
    loadsyn2, loadsyn3, plotroc, plot_network_weights
    from kalderstam.util.filehandling import parse_file, save_network, load_network
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    from random import uniform
    import time
    import numpy
    import matplotlib.pyplot as plt
    from kalderstam.neural.activation_functions import linear
    #from kalderstam.neural.training_functions import train_committee, traingd_block, train_evolutionary

    #numpy.seterr(all = 'raise')

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

    p = 4 #number of input covariates

    net = build_feedforward(p, 8, 1, output_function = linear(1))

    filename = '/home/gibson/jonask/my_tweaked_fake_data_no_noise.txt'
    #filename = '/home/gibson/jonask/my_tweaked_fake_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_no_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_with_noise.txt'
    #filename = '/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_no_noise.txt'

    P, T = parse_file(filename, targetcols = [4], inputcols = [0, 1, 2, 3], ignorecols = [], ignorerows = [], normalize = False)
    #P = P[:100,:]
    #T = T[:100, :]

    timeslots = generate_timeslots(P, T)

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
    from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
    loadsyn2, loadsyn3, plotroc, plot_network_weights
    from kalderstam.util.filehandling import parse_file, save_network
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    from random import uniform
    import time
    import numpy
    import matplotlib.pyplot as plt
    from kalderstam.neural.activation_functions import linear

    import pstats, cProfile

    #numpy.seterr(all = 'raise')
    logging.basicConfig(level = logging.INFO)
    glogger.setLoggingLevel(glogger.debug)

    cProfile.runctx("test()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

    #test()
    #plt.show()
