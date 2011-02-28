#Defines the node, and the network
from random import uniform
import numpy
import logging
from kalderstam.neural.activation_functions import linear, logsig, tanh

logger = logging.getLogger('kalderstam.neural.network')

def build_feedforward(input_number = 2, hidden_number = 2, output_number = 1, hidden_function = tanh(), output_function = logsig()):
    net = network()
    net.num_of_inputs = input_number
    inputs = range(input_number)
    
    #Hidden layer
    for i in range(int(hidden_number)):
        hidden = node(hidden_function)
        hidden.connect_nodes(inputs)
        net.hidden_nodes.append(hidden)
        
    #Output nodes
    for i in range(int(output_number)):
        output = node(output_function)
        output.connect_nodes(net.hidden_nodes)
        net.output_nodes.append(output)
    
    return net

class network:
    
    def __init__(self):
        self.num_of_inputs = 0
        self.hidden_nodes = []
        self.output_nodes = []
            
    def get_all_nodes(self):
        """Returns all nodes."""
        result_set = []
        result_set += self.hidden_nodes
        result_set += self.output_nodes
        return result_set
            
    def __len__(self):
        """The length of the network is defined as: input nodes + hidden nodes + output nodes."""
        return self.num_of_inputs + len(self.hidden_nodes) + len(self.output_nodes)
    
    def sim(self, input_array):
        return numpy.array([self.update(input) for input in input_array])
            
    #inputs is a list that must match in length with the number of input nodes
    def update(self, inputs):
        """Returns a numpy array of output value arrays."""
        if self.num_of_inputs != len(inputs):
            logger.error('Incorrect number of inputs(' + str(len(inputs)) + '), correct number is ' + str(self.num_of_inputs))
        else:
            return numpy.array([output_node.output(inputs) for output_node in self.output_nodes])

class node:
    def __int__(self):
        raise ValueError
    
    #default function is F(x) = x
    def __init__(self, active = linear(), bias = None, random_range = 1):
        self.random_range = random_range
        self.weights = {}
        self.function = active #Used to save to file
        self.activation_function = active.function
        self.activation_derivative = active.derivative
        #initialize the bias
        if bias:
            self.bias = bias
        else:
            self.bias = uniform(-self.random_range, self.random_range)
        
    def connect_node(self, node, weight = None):
        if not weight:
            weight = uniform(-self.random_range, self.random_range)
        self.weights[node] = weight
        
    def connect_nodes(self, nodes, weight_dict = None):
        for node in nodes:
            if not weight_dict:
                self.weights[node] = uniform(-self.random_range, self.random_range)
            else:
                self.weights[node] = weight_dict[node]
                
    def input_sum(self, inputs):
        input_sum = self.bias
        for node, weight in self.weights.items():
            try:
                index = int(node)
                input_sum += weight * inputs[index]
            except ValueError:
                input_sum += node.output(inputs) * weight
                
        return input_sum
        
    def output(self, inputs):
        return self.activation_function(self.input_sum(inputs))
                

if __name__ == '__main__': 
    net = build_feedforward(output_number = 2)
     
    results = net.update([1, 2])
     
    print(results)
     
    results2 = net.sim([[1, 2], [2, 3]])
    print(results2)
