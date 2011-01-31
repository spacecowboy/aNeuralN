#Defines the node, and the network
from random import uniform
from math import exp
from math import tanh
import numpy
import logging
import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger('kalderstam.neural.network')

#A few activation functions
def logsig(x):
    return 1 / (1 + exp(-x))

def linear(x):
    return x

class network:
    #Single hidden layer ANN. Weights and bias are initialized to random numbers between -1 and 1
    def build_feedforward(self, input_number=2, hidden_number=5, output_number=1, hidden_function=tanh, output_function=logsig):
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
        """Returns all nodes except the input nodes."""
        result_set = []
        result_set += self.hidden_nodes
        result_set += self.output_nodes
        return result_set
            
    def __len__(self):
        """The length of the network is defined as: input nodes + hidden nodes + output nodes."""
        return len(self.input_nodes) + len(self.hidden_nodes) + len(self.output_nodes)
    
    def sim(self, input_array):
        results = numpy.array([])
        for i in range(0, len(input_array)):
            input = input_array[i]
            results = numpy.append(results, self.update(input))
            #results = numpy.append(results, [self.update(input)], 0)
        return results
            
    #inputs is a list that must match in length with the number of input nodes
    def update(self, inputs):
        """Returns a numpy array of output value arrays."""
        if len(self.input_nodes) != len(inputs):
            logger.error('Incorrect number of inputs(' + str(len(inputs)) + '), correct number is', len(self.input_nodes))
        else:
            results = numpy.array([])
            #Update input nodes to the values
            index = 0
            for value in inputs:
                self.input_nodes[index].value = value
                index += 1
            
            for output_node in self.output_nodes:
                results = numpy.append(results, output_node.output())
            
            return results
        
    def traingd(self, input_array, output_array, epochs=300, learning_rate=0.1):
        """Train using Gradient Descent."""
        
        error_sum = 0
        for j in range(0, epochs):
            #Iterate over training data
            logger.debug('Epoch ' + str(j))
            for i in range(0, len(input_array)):
                input = input_array[i]
                # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                output = numpy.append(numpy.array([]), output_array[i])
                    
                #Calc output
                result = self.update(input)
                    
                #Set error to 0 on all nodes first
                for node in self.get_all_nodes():
                    node.error = 0
                
                #Set errors on output nodes first
                error_sum = 0
                for output_index in range(0, len(self.output_nodes)):
                    self.output_nodes[output_index].set_error(output[output_index] - result[output_index])
                    error_sum += abs(output[output_index] - result[output_index])
                    
                #Iterate over the nodes and correct the weights
                for node in self.output_nodes + self.hidden_nodes:
                    for back_node, back_weight in node.weights.iteritems():
                        back_node.error += back_weight * node.error
                        node.weights[back_node] = back_weight + learning_rate * node.error * back_node.output()

            logger.debug("Error = " + str(error_sum))

        
    def train(self, input_array, output_array, weight_calculator=lambda input_value, old_weight, error: old_weight + 0.1 * error * input_value, error_calculator=lambda weight, error: weight * error):
        if not len(input_array) == len(output_array):
            logger.error('Error: Length of input and output arrays do not match.')
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
                    self.output_nodes[j].set_error(output[j] - result[j])
                
                #Iterate over the nodes and correct the weights
                #for node in self.output_nodes + self.hidden_nodes:
                    #node.update_weights(weight_calculator, error_calculator)
            
#Special type of node, since it is really just a scalar.
class input_node:
    def __init__(self, value=1):
        self.value = value
        self.error = 0 #just here to make the back-propagation algorithm easy
    
    def output(self):
        return self.value

class node:
    #default function is F(x) = x
    def __init__(self, active=lambda x: x, active_prime=lambda x: 1):
        self.weights = dict()
        self.activation_function = active
        self.activation_derivative = active_prime
        #initialize the bias
        self.bias = uniform(-1, 1)
        #local error is zero to begin with
        self.error = 0
        
    def connect_nodes(self, nodes):
        for node in nodes:
            self.weights[node] = uniform(-1, 1)
            
    def set_error(self, delta):
        """Given a delta = (d_i(m) - y_i(m)), it is multiplied by the derivative of the activation function: Phi'(input_sum)"""
        self.error = delta * self.activation_derivative(self.input_sum)
        
    def output(self):
        self.input_sum = self.bias
        for node, weight in self.weights.iteritems():
            self.input_sum += node.output()*weight
        
        return self.activation_function(self.input_sum)
                

#if __name__ == '__main__':
#    #Binary activation function
#    def activation_function(x):
#        if x > 0:
#            return 1
#        else:
#            return 0
#                
#    net = network()
#    net.build_feedforward(2, 1, 1, output_function = activation_function)
#    
#    #T = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#    #P1 = [2.04699421438210, 0.386227934213871, 0.366680687337705, 0.583651729841386, -0.646548113409711, 0.744485584959789, 0.800551621960988, -0.0655744844339043, 0.512110467987089, 1.44050697067342, 0.284789001890266, 1.59319716783919, 0.399996113789293, 2.39006664827929, 1.53791915841959, 0.953808369089956, 1.36045712873556, 0.797762159917227, 0.579996927509666, 0.440625861574360, 1.40507910582573, 1.07160956531203, 1.24576460068840, 0.121346362479676, 1.09664386091711, 0.0763973669460567, 0.323185050093574, 0.312906907310133, -0.438248158570272, 1.39019350492901, 1.60463537979849, 0.672222508659518, 0.844883271482691, 1.36204690575305, 1.31579887067303, 1.03140909309631, 0.908980229634832, 0.0575116830765665, 0.676192872233737, 2.13521286298613, 0.623494772409942, 2.12767265739104, 0.241280368088821, 0.383379239572360, 0.00827865053454496, 0.675635199879619, 0.494655090812019, 1.90370871908190, 0.511617475492609, -0.116268026118420, 0.458025969725571, -1.23334784533326, -1.17123009327634, 0.953004086148303, -1.44281910049472, -0.520502579866877, -0.153790049603719, -0.766751282914447, -0.0221207573206125, -0.277964987163300, -2.72162310943103, -1.53608543258090, -1.11745991714396, 0.139356042341599, -0.488729859598146, -0.00333603559073001, -0.623700921909197, -2.37102006934499, 0.593039819251113, 0.0269183512767745, -0.418822348000131, -0.219944189949409, 0.386019134699817, -0.885178791784036, -0.0883619278496028, -0.229674682850103, -0.461885038047083, -1.42335625158464, -1.61425689688991, -0.691661369921274, 0.966939332851572, -0.0442941913626650, -1.22734940941890, -1.17612514969461, 0.0215015059821598, 0.345947398081640, -0.639812149086709, -0.442502695478462, 0.481846105000130, -0.236208219961442, 1.02951264872338, -1.24025119439177, -0.627288271112161, -1.44138479635650, -0.567295209615143, -0.123756016046696, -1.25889421832114, -1.16182194445294, -0.740062469645386, -0.525200244124737]
#    #P2 = [0.588968394134430, -1.30442029110504, -0.986668414423908, -0.807517052346454, -2.02967697573355, -0.591890816749359, -0.621738858204232, -1.21794123244885, -0.798977831306649, -0.0926259090218340, -0.921573260413688, -0.147487295737448, -0.869898398216369, 0.509927243322294, 0.186746064800257, -0.478129857306275, 0.00826895504469390, -0.862273254818482, -0.293754285443248, -1.33874672347079, -0.299781122437434, -0.589257107402507, -0.321926614055064, -0.809925887862396, 0.155484587200465, -0.633199394815055, -1.59817427234791, -0.761730683906530, -1.66931110010956, -0.164330148839183, 0.116743788369093, -0.573150178400562, -0.451916729692174, -0.145960982963549, -0.108212277086507, -0.763739907648170, -0.476315211263038, -1.04852041849796, -1.11095909051932, 0.736065487045927, -0.787932388391425, 0.532494699078834, -1.20658260594498, -1.13614893911507, -1.64861758293175, -0.254632759187694, -0.662753716462106, 0.721200917331614, -1.29685461039494, -1.32737780292465, 1.34641149675006, 0.153620835504997, 0.0734218481451303, 2.23448844805720, 0.193860844062950, 0.658327351598072, 1.32616775415013, 0.599764413235826, 1.33999827798811, 0.995525105778999, -0.291078521651080, -0.0850617294194311, -0.0617598507745599, 1.55544894476186, 1.04684063332315, 1.63841867738291, 0.775055380228918, -0.332528694606128, 1.78778017145148, 0.857968307261032, 0.762809917281964, 1.08018633374174, 1.92788171342576, 1.16291958183443, 0.929402641008883, 1.16004029984051, 0.611995238145295, -0.0219849038979159, 0.119262598017907, 0.867824378866351, 2.28133511307920, 1.13302900428927, 0.520144879308346, 0.204219115243884, 1.43396156440997, 1.36710106838240, 0.546911098994571, 0.722989198486450, 1.92132734728197, 1.13807997316966, 2.87616770264036, 0.101501969661722, 1.10394250011935, 0.0877545350168186, 1.10935254697210, 1.31466903209366, -0.0562191032658364, 0.100442317240541, 0.361692281537002, 0.510596351785934]
#    #P = []
#    #for count in range(0, len(P1)):
#    #    P.append([P1[count], P2[count]])
#    #    T[count] = [T[count]]
#    
#    P, T = loadsyn1(100)
#    
#    net.traingd(P, T, 10, 0.1)
#    
#    plot2d2c(net, P, T)
    
