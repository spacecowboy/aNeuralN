from kalderstam.neural.error_functions.cox_error import derivative, calc_beta, \
    calc_sigma, total_error, get_risk_groups, get_beta_force
from kalderstam.util.decorators import benchmark
import logging
from numpy import exp
import numpy as np
import kalderstam.util.graphlogger as glogger
from kalderstam.neural.training.gradientdescent import traingd
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
    if prev_timeslots_network is None:
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
    initial_minima = False
    weight_corrections = {}
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

        if prev_error is None:
            prev_error = current_error
        elif current_error <= prev_error:
            prev_error = current_error
            initial_minima = False #We must be out of it
        elif initial_minima and current_error > prev_error:
            prev_error = current_error
            logger.info('Allowing initial climb') #It's ok
        elif corrected:
            #Undo the weight correction
            apply_weight_corrections(net, -learning_rate, weight_corrections)
            corrected = False
            #Half the learning rate
            learning_rate *= 0.5
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

        weight_corrections = {}
        #Set corrections to 0 on all nodes first
        for node in net.get_all_nodes():
            weight_corrections[node] = {}

        #Iterate over all output indices
        i = 0
        for input, output_index in zip(inputs, range(len(outputs))):
            logger.debug("Patient: " + str(i))
            i += 1
            #Set error to 0 on all nodes first
            gradients = {}
            for node in net.get_all_nodes():
                gradients[node] = 0

            #Set errors on output nodes first
            for node, gradient in zip(net.output_nodes, derivative(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)):
                #glogger.debugPlot('Gradient', gradient, style = 'b.')
                gradients[node] = gradient

            #Iterate over the nodes and correct the weights
            for node in net.output_nodes + net.hidden_nodes:
                #Calculate local error gradient
                gradients[node] *= node.output_derivative(input)

                #Propagate the error backwards and then update the weights
                for back_node, back_weight in node.weights.items():

                    if back_node not in weight_corrections[node]:
                        weight_corrections[node][back_node] = []

                    try:
                        index = int(back_node)
                        weight_corrections[node][back_node].append(gradients[node] * input[index])
                    except TypeError:
                        gradients[back_node] += back_weight * gradients[node]
                        weight_corrections[node][back_node].append(gradients[node] * back_node.output(input))

                #Finally, correct the bias
                if "bias" not in weight_corrections[node]:
                    weight_corrections[node]["bias"] = []
                weight_corrections[node]["bias"].append(gradients[node])

        apply_weight_corrections(net, learning_rate, weight_corrections)
        corrected = True

    try:
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
    except FloatingPointError as e:
        print(str(e))

    sigma = calc_sigma(outputs)

    current_error = total_error(beta, sigma)

    if prev_error is None:
        prev_error = current_error
    elif current_error <= prev_error:
        initial_minima = False #We must be out of it
    elif initial_minima and current_error > prev_error:
        pass #It's ok
    elif corrected:
        #Undo the weight correction
        apply_weight_corrections(net, -learning_rate)

    return net

def apply_weight_corrections(net, learning_rate, weight_corrections):
    #Iterate over the nodes and correct the weights
    for node in net.output_nodes + net.hidden_nodes:
        #Calculate weight update
        for back_node, back_weight in node.weights.items():
            node.weights[back_node] = back_weight - learning_rate * sum(weight_corrections[node][back_node]) / len(weight_corrections[node][back_node])
        #Don't forget bias
        node.bias = node.bias - learning_rate * sum(weight_corrections[node]["bias"]) / len(weight_corrections[node]["bias"])
