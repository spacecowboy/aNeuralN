#Defines the node, and the network
from random import uniform
import numpy
import logging
from kalderstam.neural.activation_functions import linear, logsig, tanh
#from kalderstam.util.exceptions import ArgumentError
from multiprocessing import Pool
import os
from numpy.oldnumeric.random_array import ArgumentError
import time

logger = logging.getLogger('kalderstam.neural.network')


def build_feedforward(input_number = 2, hidden_number = 2, output_number = 1, hidden_function = tanh(), output_function = logsig(), random_range=1.0):
    net = network(input_number = input_number, hidden_number = hidden_number, output_number = output_number)
    
    #Hidden layer
    for i in range(int(hidden_number)):
        weights = numpy.zeros(len(net))
        for input in range(input_number):
            weights[input] = uniform(-random_range, random_range)
        #bias
        weights[input_number + i] = uniform(-random_range, random_range)
        net.set_weights(input_number + i, weights)
        
    #Output nodes
    for i in range(int(output_number)):
        weights = numpy.zeros(len(net))
        for hidden in range(input_number, input_number + hidden_number):
            weights[hidden] = uniform(-random_range, random_range)
        #bias
        weights[input_number + hidden_number + i] = uniform(-random_range, random_range)
        net.set_weights(input_number + hidden_number + i, weights)
    
    return net

def pad_input(net, input):
    if len(input) == len(net):
        return input
    else:
        result = numpy.ones(len(net))
        for i in range(len(input)):
            result[i] = input[i]
        return result

class network:
    
    def __init__(self, input_number = 2, hidden_number = 2, output_number = 1, activation_functions = None):
        self.num_of_inputs = input_number
        self.num_of_outputs = output_number
        cols = input_number + hidden_number + output_number
        self.output_start = hidden_number #The row where the first output node resides
        self.weights = numpy.identity(cols)
        #self.weights = numpy.delete(self.weights, range(input_number), 0) #Remove input rows as they will be all zeroes anyway
        
        if activation_functions != None:
            self.activation_functions = numpy.array(activation_functions)
        else:
            t = tanh()
            l = logsig()
            self.activation_functions = [None for i in range(input_number)] # Is never used
            [self.activation_functions.append(t) for i in range(hidden_number)]
            [self.activation_functions.append(l) for i in range(output_number)]
            self.activation_functions = numpy.array(self.activation_functions)
            
    def set_weights(self, node, weights):
        self.weights[node] = weights
        
    def fix_layers(self):
        """Calculates what layers the nodes belong in."""
        #self.layers.append([]) #input nodes are layer 0, are not used for calculations
        distances = [0 for n in range(len(self))] #initialize vector
        for node_index in range(self.num_of_inputs, len(self.weights)):
            for weight, weight_index in zip(self.weights[node_index], range(len(self.weights[node_index]))):
                if weight_index != node_index and weight != 0:
                    distances[node_index] = max(distances[node_index], distances[weight_index] + 1)
                    
        self.layers = [[] for d in range(max(distances))]
        for d, weight_index in zip(distances, range(len(distances))):
            if d > 0:
                self.layers[d-1].append(weight_index)
            
    def __len__(self):
        """The length of the network is defined as: input nodes + hidden nodes + output nodes."""
        return len(self.weights[0])
    
    def __check_input(self, input):
        if len(input) != len(self):
            raise ArgumentError("Input array was not of correct length: " + str(len(input)) + " vs " + str(len(self)))
    
    def sim(self, input_array):
        """For a list of input_values, outputs a list of output values. Also modifies input_array so that contains all information as well."""
        return numpy.array([self.update(input) for input in input_array])
            
    #inputs is an array that must match in length with the number of nodes.
    def update(self, inputs):
        """Returns a numpy array of output values. Input vector is also modified! And contains this information as well"""
        self.__check_input(inputs)
        for rows in self.layers: #Traverse the network
            inputs[rows] = pool.map(mp_calc_outputs, [(self.weights[row], inputs, self.activation_functions[row]) for row in rows])
            #input_sum = numpy.dot(self.weights[rows], inputs)
            #map_result = pool.map(mp_calc_outputs, [(self.weights[row], inputs, self.activation_functions[row]) for row in rows])
            #inputs[rows] = [active.function(input) for active, input in zip(self.activation_functions[rows], input_sum)]
        #Now return output values stored in input vector
        return numpy.delete(inputs, range(self.output_start + self.num_of_inputs))
    
def mp_calc_outputs((weights, inputs, activation_function)):
    """For multiprocessing map, only supports one iterable argument."""
    input_sum = numpy.dot(weights, inputs)
    value =  activation_function.function(input_sum)
    #Wait one second...
    #t = time.time()
    #while time.time() - t < 1:
    #    pass
    #print 'process id:', os.getpid(), 'input_sum:', input_sum, 'value:', value
    return value


pool = Pool() #Processing pool, default to number of cpu-cores

if __name__ == '__main__': 
    net = build_feedforward(input_number = 8, hidden_number = 8, output_number = 8)
    net.fix_layers()
    
    i = pad_input(net, [i for i in range(8)])
    start = time.time()
    for times in range(1):
        results = net.update(i)
    stop = time.time()
    
    
    #i = pad_input(net, [1, 2])
    #results = net.update(i)
    #print(i)
     
    print(results)
    print('Time was: ', stop - start)
     
    #i2 = [pad_input(net, [1, 2]), pad_input(net, [2, 3])]
    #results2 = net.sim(i2)
    #print(i2)
    #print(results2)
    pool.close()
