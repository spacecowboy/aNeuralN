from kalderstam.neural.error_functions import sum_squares
from kalderstam.neural.network import node
from random import sample
import kalderstam.util.graphlogger as glogger
import logging
import numpy as np

logger = logging.getLogger('kalderstam.neural.training.gradientdescent')
np.seterr(all = 'raise') #I want errors!

def traingd(net, (test_inputs, test_targets), (validation_inputs, validation_targets), epochs = 300, learning_rate = 0.1, block_size = 1, momentum = 0.0, error_derivative = sum_squares.derivative, error_function = sum_squares.total_error, stop_error_value = 0, pre_loop_func = None, epoch_func = None, block_func = None):
    """Train using Gradient Descent.
    The pre_loop function calculcates possible values that are necessary for the error function later on (for performance reasons probably).
    It should return a dict which will be passed as keyword arguments to the other functions.
    Same for epoch_func and block_func."""

    if pre_loop_func:
        pre_loop_kwargs = pre_loop_func(net, (test_inputs, test_targets), (validation_inputs, validation_targets))
    else:
        pre_loop_kwargs = {}

    for epoch in range(0, int(epochs)):
        try: #Want to catch keyboard interrupt
            #Iterate over training data
            logger.info('Epoch ' + str(epoch))
            if epoch_func:
                epoch_kwargs = epoch_func(net, (test_inputs, test_targets), (validation_inputs, validation_targets), pre_loop_kwargs)
            else:
                epoch_kwargs = {}
            #error_sum = 0
            block_size = int(block_size)
            if block_size < 1 or block_size > len(test_inputs): #if 0, then equivalent to batch. 1 is equivalent to online
                block_size = len(test_inputs)

            for block in range(int(len(test_inputs) / block_size)):

                weight_corrections = {}
                gradients = {}
                #Set corrections to 0 on all nodes first
                for node in net.get_all_nodes():
                    weight_corrections[node] = {}

                #Train in random order
                block_data = sample(zip(test_inputs, test_targets), block_size)

                if block_func:
                    block_kwargs = block_func(net, (test_inputs, test_targets), (validation_inputs, validation_targets), pre_loop_kwargs, epoch_kwargs)
                else:
                    block_kwargs = {}

                for input, target in block_data:

                    #Calc output
                    result = net.update(input)

                    #Set error to 0 on all nodes first
                    for node in net.get_all_nodes():
                        gradients[node] = 0

                    #Set errors on output nodes first
                    extra_kwargs = dict(pre_loop_kwargs.items() + epoch_kwargs.items() + block_kwargs.items())
                    for node, gradient in zip(net.output_nodes, error_derivative(target, result, **extra_kwargs)):
                        glogger.debugPlot('Gradient at output nodes', gradient, style = 'b-')
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

            #Calculate error of the network and print

            if len(test_inputs > 0):
                test_results = net.sim(test_inputs)
                test_error = error_function(test_targets, test_results) / len(test_targets)
                glogger.debugPlot('Test Error', test_error, style = 'r-')
                logger.debug("Test Error = " + str(test_error))
                if test_error <= stop_error_value:
                    break

            if validation_inputs != None and len(validation_inputs) > 0:
                validation_results = net.sim(validation_inputs)
                validation_error = error_function(validation_targets, validation_results) / len(validation_targets)
                logger.debug("Validation Error = " + str(validation_error))
                if validation_error <= stop_error_value:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupt received, returning net...")
            break

    return net

def apply_weight_corrections(net, learning_rate, weight_corrections):
    #Iterate over the nodes and correct the weights
    for node in net.output_nodes + net.hidden_nodes:
        #Calculate weight update
        for back_node, back_weight in node.weights.items():
            node.weights[back_node] = back_weight - learning_rate * sum(weight_corrections[node][back_node]) / len(weight_corrections[node][back_node])
        #Don't forget bias
        node.bias = node.bias - learning_rate * sum(weight_corrections[node]["bias"]) / len(weight_corrections[node]["bias"])
