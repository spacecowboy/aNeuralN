#Defines the node, and the network
from random import uniform
import numpy
import logging
from kalderstam.neural.activation_functions import linear, logsig, tanh

logger = logging.getLogger('kalderstam.neural.network')

def build_feedforward(input_number = 2, hidden_number = 2, output_number = 1, hidden_function = tanh(), output_function = logsig(), random_range=1.0):
    net = network(input_number = input_number, hidden_number = hidden_number, output_number = output_number, hidden_function = hidden_function, output_function = output_function)
    
    #Hidden layer
    for i in range(int(hidden_number)):
        weights = numpy.zeros(len(net))
        for input in range(input_number):
            weights[input] = uniform(-random_range, random_range)
        #bias
        weights[input_number + i] = uniform(-random_range, random_range)
        net.set_weights(i, weights)
        
    #Output nodes
    for i in range(int(output_number)):
        weights = numpy.zeros(len(net))
        for hidden in range(input_number, input_number + input_number):
            weights[hidden] = uniform(-random_range, random_range)
        #bias
        weights[input_number + hidden_number + i] = uniform(-random_range, random_range)
        net.set_weights(hidden_number + i, weights)
    
    return net

class network:
    
    def __init__(self, input_number = 2, hidden_number = 2, output_number = 1, hidden_function = linear(), output_function = linear()):
        self.num_of_inputs = input_number
        self.num_of_outputs = output_number
        cols = input_number + hidden_number + output_number
        self.hidden_function = hidden_function
        self.output_function = output_function
        self.output_start = hidden_number #The row where the first output node resides
        self.weights = numpy.identity(cols)
        self.weights = numpy.delete(self.weights, range(input_number), 0) #Remove input rows as they will be all zeroes anyway
            
    def set_weights(self, node, weights):
        self.weights[node] = weights
        
    def fix_layers(self):
        """Calculates what layers the nodes belong in."""
        #self.layers.append([]) #input nodes are layer 0, are not used for calculations
        distances = [0 for n in range(len(self))] #initialize vector
        for node_index in range(len(self.weights)):
            for weight, weight_index in zip(self.weights[node_index], range(len(self.weights[node_index]))):
                if weight_index != node_index + self.num_of_inputs and weight != 0:
                    distances[node_index + self.num_of_inputs] = distances[weight_index] + 1
                    
        self.layers = [[] for d in range(max(distances))]
        for d, weight_index in zip(distances, range(len(distances))):
            if d > 0:
                self.layers[d-1].append(weight_index - self.num_of_inputs)
            
    def __len__(self):
        """The length of the network is defined as: input nodes + hidden nodes + output nodes."""
        return len(self.weights[0])
    
    def __pad_input(self, input):
        if len(input) == len(self):
            return input
        else:
            result = numpy.ones(len(self))
            for i in range(len(input)):
                result[i] = input[i]
            return result
    
    def sim(self, input_array):
        input_array = numpy.array([self.__pad_input(input) for input in input_array])
        return numpy.array([self.update(input) for input in input_array])
            
    #inputs is an array that must match in length with the number of nodes.
    def update(self, inputs):
        """Returns a numpy array of output value arrays."""
        col = self.num_of_inputs
        inputs = self.__pad_input(inputs)
        for rows in self.layers: #Traverse the network
            input_sum = numpy.dot(self.weights[rows], inputs)
            if row < self.output_start:
                inputs[rows + self.num_of_inputs] = self.hidden_function.function(input_sum)
            else:
                inputs[col] = self.output_function.function(input_sum)
            col += 1
        #Now return output values stored in input vector
        return numpy.delete(inputs, range(self.output_start + self.num_of_inputs))

if __name__ == '__main__': 
    net = build_feedforward(input_number = 2, hidden_number = 2, output_number = 2)
    net.fix_layers()
    
    results = net.update([1, 2])
     
    print(results)
     
    results2 = net.sim([[1, 2], [2, 3]])
    print(results2)
