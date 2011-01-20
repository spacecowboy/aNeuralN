#Defines the node, and the network
from math import exp
from random import uniform

class network:
    #Single hidden layer ANN. Weights and bias are initialized to random numbers between -1 and 1
    def build_feedforward(self, input_number=2, hidden_number=2, output_number=1, hidden_function=lambda x: x, output_function=lambda x: x):
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        
        #Input nodes
        for i in range(input_number):
            input = input_node()
            self.input_nodes.append(input)
        
        #Hidden layer
        for i in range(hidden_number):
            hidden = node(hidden_function)
            hidden.connect_nodes(self.input_nodes)
            self.hidden_nodes.append(hidden)
            
        #Output nodes
        for i in range(output_number):
            output = node(output_function)
            output.connect_nodes(self.hidden_nodes)
            self.output_nodes.append(output)
            
    def get_all_nodes(self):
        result_set = []
        result_set += self.hidden_nodes
        result_set += self.output_nodes
        return result_set
            
    def __len__(self):
        return len(self.input_nodes) + len(self.hidden_nodes) + len(self.output_nodes)
            
    #inputs is a list that must match in length with the number of input nodes
    def update(self, inputs):
        if not len(self.input_nodes) == len(inputs):
            print 'Incorrect number of inputs(' + str(len(inputs)) + '), correct number is', len(self.input_nodes)
        else:
            results = []
            #Update input nodes to the values
            index = 0
            for value in inputs:
                self.input_nodes[index].value = value
                index += 1
            
            for output_node in self.output_nodes:
                results.append(output_node.output())
                
            return results
        
    def train(self, input_array, output_array, weight_calculator=lambda input_value, old_weight, error: old_weight + 0.1 * error * input_value, error_calculator=lambda weight, error: weight * error):
        if not len(input_array) == len(output_array):
            print 'Error: Length of input and output arrays do not match'
        else:
            #Iterate over training data
            for i in range(0, len(input_array)):
                input = input_array[i]
                output = output_array[i]
                
                #Calc output
                result = self.update(input)
                
                #Set error to 0 on all nodes first
                for node in self.get_all_nodes():
                    node.error = 0
                
                #Set errors on output nodes first
                for j in range(0, len(self.output_nodes)):
                    self.output_nodes[j].error = output[j] - result[j]
                
                #Iterate over the nodes and correct the weights
                for node in self.output_nodes + self.hidden_nodes:
                    node.update_weights(weight_calculator, error_calculator)
            

class input_node:
    def __init__(self, value=1):
        self.value = value
        self.error = 0 #just here to make the back-propagation algorithm easy
    
    def output(self):
        return self.value

class node:
    #default function is F(x) = x
    def __init__(self, func=lambda x: x):
        self.weights = dict()
        self.connected_nodes = []
        self.activation_function = func
        #initialize the bias
        self.bias = uniform(-1, 1)
        #local error is zero to begin with
        self.error = 0
        
    def connect_nodes(self, nodes):
        self.connected_nodes += nodes
        for node in nodes:
            self.weights[node] = uniform(-1, 1)
            
    #The weight calculator must take the arguments (input, weight)
    #The error calculator must take the arguments (weight, error)
    def update_weights(self, weight_calculator, error_calculator):
        for node, weight in self.weights.iteritems():
            #print 'Weight correction = ' + str((weight - weight_calculator(node.output(), weight)))
            node.error += error_calculator(weight, self.error)
            self.weights[node] = weight_calculator(node.output(), weight, self.error)
        
    def output(self):
        self.input_sum = self.bias
        for node in self.connected_nodes:
            self.input_sum += node.output()*self.weights[node]
        
        return self.activation_function(self.input_sum)

if __name__ == '__main__':
    #Binary activation function
    def activation_function(x):
        if x > 0:
            return 1
        else:
            return - 1
                
    net = network()
    net.build_feedforward(2, 2, 1, output_function=activation_function)
    #XOR-values
    data = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    answers = [[-1], [1], [1], [-1]]
    for j in range(0, 500):
        print "Iteration " + str(j) + "\n"
        net.train(data, answers)
        
        for input in data:
            net.update(input)
                
            print "Input: " + str(input)
            i = 0
            for node in net.get_all_nodes():
                print "Node " + str(i) + ": Weights = " + str([node.bias] + node.weights.values()) + ": Output = " + str(node.output())
                i += 1
            print "\n"
    
