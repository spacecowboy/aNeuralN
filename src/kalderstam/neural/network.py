#Defines the node, and the network
from random import uniform
import numpy
import logging
from kalderstam.neural.activation_functions import linear, logsig, tanh
from kalderstam.util.decorators import benchmark

logger = logging.getLogger('kalderstam.neural.network')

def build_feedforward_committee(size = 8, input_number = 2, hidden_number = 2, output_number = 1, hidden_function = tanh(), output_function = logsig()):
    net_list = [build_feedforward(input_number, hidden_number, output_number, hidden_function, output_function) for n in range(size)]
    return committee(net_list)

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

class committee:
    def __init__(self, net_list = None):
        if net_list == None:
            self.nets = []
        else:
            self.nets = net_list

    def __len__(self):
        return len(self.nets)

    def update(self, inputs):
        results = [net.update(inputs) for net in self.nets]
        return self.__average__(results)

    def sim(self, input_array):
        return numpy.array([self.update(input) for input in input_array])

    def __average__(self, outputs):
        """Outputs is a list of network outputs. Each a list of output node results."""
        result = outputs[0] - outputs[0] #A zero array of the same shape as output
        #Calculate average
        for output in outputs: #Sum all values
            result += output
        result /= len(self) #Divide by size
        return result #Returns an array of average values for each output node

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

    def __mul__(self, number):
        """Multiplying a network with a number is the same as multiplying all weights with that number.
        Note, this happens in place and changes the network you're multiplying with"""
        number = float(number) #If number doesn't have a floating point representation, this will throw an exception
        for node in self.get_all_nodes():
            for connected_node in node.weights.keys():
                node.weights[connected_node] = number * node.weights[connected_node]
        return self

    def __div__(self, number):
        """Multiplying a network with a number is the same as multiplying all weights with that number."""
        return self.__mul__(1 / number)

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

    #default activation_function is F(x) = x
    def __init__(self, active = linear(), bias = None, random_range = 1):
        self.random_range = random_range
        self.weights = {}
        self.activation_function = active #Used to save to file
        #self.activation_function = active.activation_function
        #self.activation_derivative = active.derivative
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
        return self.activation_function.function(self.input_sum(inputs))

    def output_derivative(self, inputs):
        return self.activation_function.derivative(self.input_sum(inputs))
