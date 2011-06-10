from kalderstam.neural.error_functions import sum_squares
from kalderstam.neural.network import node
from random import sample
import kalderstam.util.graphlogger as glogger
import logging
import numpy as np

logger = logging.getLogger('kalderstam.neural.training.gradientdescent')
np.seterr(all = 'raise') #I want errors!

def traingd(net, (test_inputs, test_targets), (validation_inputs, validation_targets), epochs = 300, learning_rate = 0.1, block_size = 1, error_module = sum_squares):
    """Train using Gradient Descent.
    The pre_loop function calculcates possible values that are necessary for the error function later on (for performance reasons probably).
    It should return a dict which will be passed as keyword arguments to the other functions.
    Same for epoch_func and block_func."""

    block_size = int(block_size)
    if block_size < 1 or block_size > len(test_inputs): #if 0, then equivalent to batch. 1 is equivalent to online
        block_size = len(test_inputs)

    extra_kwargs = {}
    if hasattr(error_module, 'pre_loop_func'):
        pre_loop_kwargs = error_module.pre_loop_func(net, test_inputs, test_targets, block_size)
    else:
        pre_loop_kwargs = {}

    pre_error, error = None, None

    for epoch in xrange(0, int(epochs)):
        try: #Want to catch keyboard interrupt
            #Iterate over training data
            logger.info('Epoch ' + str(epoch))

            if hasattr(error_module, 'epoch_func'):
                epoch_kwargs = error_module.epoch_func(net, test_inputs, test_targets, block_size, **pre_loop_kwargs)
            else:
                epoch_kwargs = {}

            #For varying learning rate, calculate if the last step improved. Use only for batch.
            if block_size == len(test_inputs):
                results = net.sim(test_inputs)
                error = error_module.total_error(test_targets, results, **epoch_kwargs)
                if not pre_error:
                    pre_error = error
                elif error <= pre_error: #accept changes, increase learning rate
                    learning_rate *= 1.1
                    logger.debug('Error %1.4f <= %1.4f, learning rate set to %2.4f ', error, pre_error, learning_rate)
                    pre_error = error
                else: #Roll back
                    apply_weight_corrections(net, -learning_rate, weight_corrections) #Negative rate to reverse the changes
                    learning_rate *= 0.5 #Try with smaller learning rate
                    logger.debug('Error %1.4f > %1.4f, learning rate set to %2.4f ', error, pre_error, learning_rate)


            for block in xrange(int(len(test_inputs) / block_size)):
                results = net.sim(test_inputs)
                extra_kwargs.update(pre_loop_kwargs)
                extra_kwargs.update(epoch_kwargs) #Add data from epoch function

                weight_corrections = {}
                gradients = {}
                #Set corrections to 0 on all nodes first
                for node in net.get_all_nodes():
                    weight_corrections[node] = {}

                #Train in random order
                block_data = sample(range(len(test_targets)), block_size)

                if hasattr(error_module, 'block_func'):
                    extra_kwargs.update(error_module.block_func(test_inputs, test_targets, block_size, results, block_data, **extra_kwargs))

                for index, member_index in zip(block_data, range(len(block_data))):
                    input = test_inputs[index]
                    member_targets, member_results = test_targets[block_data], results[block_data]

                    #Set error to 0 on all nodes first
                    for node in net.get_all_nodes():
                        gradients[node] = 0

                    #Set errors on output nodes first
                    gradient = error_module.derivative(member_targets, member_results, member_index, **extra_kwargs)
                    for node in net.output_nodes:
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

            #if len(test_inputs > 0):
                #test_results = net.sim(test_inputs)
                #test_error = error_function(test_targets[:, 0], test_results[:, 0]) / len(test_targets)
                #glogger.debugPlot('Test Error', test_error, style = 'r-')
                #logger.debug("Test Error = " + str(test_error))
                #if test_error <= stop_error_value:
                    #break

            #if validation_inputs != None and len(validation_inputs) > 0:
                #validation_results = net.sim(validation_inputs)
                #validation_error = error_function(validation_targets[:, 0], validation_results[:, 0]) / len(validation_targets)
                #logger.debug("Validation Error = " + str(validation_error))
                #if validation_error <= stop_error_value:
                    #break
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
