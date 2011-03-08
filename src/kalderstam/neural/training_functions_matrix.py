import logging
import numpy
from random import sample, random, uniform
from kalderstam.neural.matrix_network import build_feedforward, network,\
    pad_input
from kalderstam.neural.error_functions import sum_squares

logger = logging.getLogger('kalderstam.neural.training_functions')

def traingd_block(net, input_array, target_array, epochs = 300, learning_rate = 0.1, block_size = 1, momentum = 0.0, error_derivative = sum_squares.derivative):
    """Train using Gradient Descent."""
    
    def node_input(node):
        input_to_node = net.weights[node]*input
        input_to_node[node] = net.weights[node, node] #Correct for bias, which must have 1 as input
        return input_to_node.sum()
    
    for epoch in range(0, int(epochs)):
        #Iterate over training data
        logger.debug('Epoch ' + str(epoch))
        #error_sum = 0
        block_size = int(block_size)
        if block_size < 1 or block_size > len(input_array): #if 0, then equivalent to batch. 1 is equivalent to online
            block_size = len(input_array)
        
        for block in range(int(len(input_array) / block_size)):
            
            weight_corrections = numpy.zeros_like(net.weights)
            
            #Train in random order
            for i, t in sample(zip(input_array, target_array), block_size):
                # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                target = numpy.append(numpy.array([]), t)

                #prepare input
                input = pad_input(net, i)
                #Calc output
                result = net.update(input)
                
                #Now, calculate weight corrections
                #First, set gradients to zero
#                error_gradients = numpy.zeros_like(net.weights[0])
#                #Then, set output gradients
#                error_gradients[net.output_layer] = error_derivative(target, result)
#                #Now, iterate over the layers backwards, updating the gradients and weight updates in the process
#                layers = list(net.layers) #Important, so we make a copy of the list. Reversing the lsit will make mayhem otherwise
#                layers.reverse()
#                """This loop is total 3 seconds of work"""
#                for rows, back_rows in zip(layers[:-1], layers[1:]): #skip the input layer
#                    """Means 0.45 seconds of work"""
#                    error_gradients[back_rows] = numpy.dot(net.weights[min(rows):(max(rows) + 1), back_rows].transpose(), error_gradients[rows])
#                    
#                    """This row is 1.2 seconds of work"""
#                    gradient = numpy.matrix([net.activation_functions[node].derivative(node_input(node)) for node in rows] * error_gradients[rows]).T.A #transposed and back to array
#                    """Bias calculation is wrong, because it doesn't remove the previous erroneous bias. just adds to it!"""
#                    """These two alone mean 1.3 seconds of work!"""
#                    weight_corrections[min(rows):(max(rows) + 1), back_rows] += learning_rate * gradient * input[back_rows]
#                    weight_corrections[min(rows):(max(rows) + 1), rows] += learning_rate * gradient * net.weights[min(rows):(max(rows) + 1), rows] #bias
                
            #Apply the weight updates
#            net.weights += weight_corrections/block_size

    return net
            

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    
    try:
        from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
        loadsyn2, loadsyn3
        from kalderstam.util.filehandling import parse_file, save_network
        import time
        import matplotlib.pyplot as plt
    except:
        pass
        
    P, T = loadsyn3(100)
    #P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Ecg1664_trn.dat", 39, ignorecols = 40)
                
    net = build_feedforward(2, 3, 1)
    net.fix_layers()
    
    epochs = 100
    
    start = time.time()
    best = traingd_block(net, P, T, epochs, block_size = 100)
    print("traingd_block time: " + str(time.time()-start))
    Y = best.sim([pad_input(best, Pv) for Pv in P])
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    #plot2d2c(best, P, T, 3)
    #plt.title("Gradient Descent block size 100\n Total performance = " + str(total_performance) + "%")
    print("\nResults for the training:\n")
    print("Total number of data: " + str(len(T)) + " (" + str(num_second) + " ones and " + str(num_first) + " zeros)")
    print("Number of misses: " + str(missed) + " (" + str(total_performance) + "% performance)")
    print("Specificity: " + str(num_correct_first) + "% (Success for class 0)")
    print("Sensitivity: " + str(num_correct_second) + "% (Success for class 1)")
    
#    start = time.clock()
#    best = train_evolutionary_sequential(net, P, T, epochs / 5, random_range = 5, block_size = 0)
#    stop = time.clock()
#    Y = best.sim(P)
#    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
#    plot2d2c(best, P, T, 4)
#    plt.title("Only Genetic Sequential\n Total performance = " + str(total_performance) + "%")
#    print("Sequential time: " + str(stop-start))
#    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/test_genetic.ann")
#    
#    #net = build_feedforward(2, 4, 1)
#    
#    best = traingd_block(best, P, T, epochs, block_size = 100)
#    Y = best.sim(P)
#    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
#    plot2d2c(best, P, T, 5)
#    plt.title("Genetic Sequential followed by Gradient Descent block size 10\n Total performance = " + str(total_performance) + "%")
    
    #plotroc(Y, T)
    #plt.show()
    
