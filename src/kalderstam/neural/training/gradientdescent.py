from kalderstam.neural.error_functions import sum_squares
from kalderstam.neural.network import node
from random import sample
import kalderstam.util.graphlogger as glogger
import logging
import numpy as np

logger = logging.getLogger('kalderstam.neural.training.gradientdescent')
np.seterr(all = 'raise') #I want errors!

def traingd(net, (test_inputs, test_targets), (validation_inputs, validation_targets), epochs = 300, learning_rate = 0.1, block_size = 1, momentum = 0.0, error_derivative = sum_squares.derivative, error_function = sum_squares.total_error, stop_error_value = 0):
    """Train using Gradient Descent."""

    for epoch in range(0, int(epochs)):
        #Iterate over training data
        logger.info('Epoch ' + str(epoch))
        #error_sum = 0
        block_size = int(block_size)
        if block_size < 1 or block_size > len(test_inputs): #if 0, then equivalent to batch. 1 is equivalent to online
            block_size = len(test_inputs)

        for block in range(int(len(test_inputs) / block_size)):

            #Set corrections to 0 on all nodes first
            for node in net.get_all_nodes():
                node.weight_corrections = {}

            #Train in random order
            for input, target in sample(zip(test_inputs, test_targets), block_size):

                #Calc output
                result = net.update(input)

                #Set error to 0 on all nodes first
                for node in net.get_all_nodes():
                    node.error_gradient = 0

                #Set errors on output nodes first
                for node, gradient in zip(net.output_nodes, error_derivative(target, result)):
                    glogger.debugPlot('Gradient at output nodes', gradient, style = 'b-')
                    node.error_gradient = gradient

                #Iterate over the nodes and correct the weights
                for node in net.output_nodes + net.hidden_nodes:
                    #Calculate local error gradient
                    node.error_gradient *= node.activation_function.derivative(node.input_sum(input))

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

    return net

def apply_weight_corrections(net, learning_rate):
    #Iterate over the nodes and correct the weights
    for node in net.output_nodes + net.hidden_nodes:
        #Calculate weight update
        for back_node, back_weight in node.weights.items():
            node.weights[back_node] = back_weight - learning_rate * sum(node.weight_corrections[back_node]) / len(node.weight_corrections[back_node])
        #Don't forget bias
        node.bias = node.bias - learning_rate * sum(node.weight_corrections["bias"]) / len(node.weight_corrections["bias"])
