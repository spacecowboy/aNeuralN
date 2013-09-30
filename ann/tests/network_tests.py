'''
Created on Apr 12, 2011

@author: jonask
'''
import unittest
#from ..network import build_feedforward
from ann.network import build_feedforward, connect_node, network, node as Node
#from ..network import build_feedforward_committee
from ann.network import build_feedforward_committee
#from kalderstam.neural.fast_network import Node


class Test(unittest.TestCase):

    def test1Simple(self):
        net = build_feedforward(input_number = 2, hidden_number = 3, output_number = 1)

        results = net.update([1, 2])
        print(results)

        results = net.sim([[1, 2], [2, 3]])
        print(results)

        com = build_feedforward_committee(input_number = 2, hidden_number = 3, output_number = 1)

        results = com.update([1, 2])
        print(results)

        results = com.sim([[1, 2], [2, 3]])
        print(results)
        print("Simple done")

    def test2Multiplication(self):
        net = build_feedforward(input_number = 2, hidden_number = 3, output_number = 1)
        first_sum = 0
        for node in net.get_all_nodes():
            for weight in node.weights.values():
                first_sum += weight
        a = -11.0
        net = net * a
        second_sum = 0
        for node in net.get_all_nodes():
            for weight in node.weights.values():
                second_sum += weight

        net / a
        third_sum = 0
        for node in net.get_all_nodes():
            for weight in node.weights.values():
                third_sum += weight

        print(first_sum, second_sum, third_sum)
        assert(round(a * first_sum, 10) == round(second_sum, 10))
        assert(round(first_sum, 10) == round(second_sum / a, 10))
        assert(round(first_sum, 10) == round(third_sum, 10))
        print("mul done")
        
    def test3Bias(self):
        net = build_feedforward(input_number = 2, hidden_number = 2, output_number = 1, hidden_function = "linear", output_function = "linear")
        
        #Bias node should always give an output of 1
        bias = net.bias_node
        
        assert(1.0 == bias.output([1,1]))
        assert(1.0 == bias.output("Crazy string argument"))
        assert(1.0 ==  bias.output(1.0))
        assert(1.0 == bias.output(-1))
        assert(1.0 == bias.output(None))
        
        inputs = [1.0,1.0]
        
        #Set all weights to 1
        for node in net.hidden_nodes:
            for key, val in node.weights.iteritems():
                node.weights[key] = 1.0
        for key, val in net.output_nodes[0].weights.iteritems():
            net.output_nodes[0].weights[key] = 1.0
        
        one_bias = net.update(inputs)
        print("Bias weight is 1: {0}".format(one_bias))
        assert(one_bias == 7.0)
        
        #Set bias weights to 0
        for node in net.hidden_nodes:
            node.weights[bias] = 0.0
        net.output_nodes[0].weights[bias] = 0.0
        
        zero_bias = net.update(inputs)
        print("Bias weight is 0: {0}".format(zero_bias))
        
        assert(one_bias > zero_bias)
        assert(zero_bias == 4.0)
        
        #Set bias weights to -99
        for node in net.hidden_nodes:
            node.weights[bias] = -99.0
        net.output_nodes[0].weights[bias] = -99.0
        
        neg_bias = net.update(inputs)
        print("Bias weight is -99: {0}".format(neg_bias))
        
        assert(zero_bias > neg_bias)
        assert(neg_bias == ((1.0*2 - 99.0)*2 - 99.0))
        
    def test4NodesTest(self):
        net = network()
        #input_number = 2 # 2 inputs
        #net.num_of_inputs = input_number
        #inputs = range(input_number)
    
        biasnode = net.bias_node
        
        inputs = range(2)
        
        my_node = Node("linear")
        for in_idx in inputs:
            #connect with inputs with weight 1.0
            connect_node(my_node, in_idx, 1.0)
            
        out_val = my_node.output([1.0,1.0])
        print("Output of node with (1,1) is: {0}".format(out_val))
        assert(out_val == 2.0) # Simply the sum of the inputs
        
        #connect to bias
        connect_node(my_node, biasnode, 1.0)
        out_val = my_node.output([1.0,1.0])
        print("Output of node with (1,1) and bias is: {0}".format(out_val))
        assert(out_val == 3.0) # Simply the sum of the inputs
        
        
    def test5NormalNodesLinear(self):
        net = build_feedforward(input_number = 2, hidden_number = 2, output_number = 1, hidden_function = "linear", output_function = "linear")
        assert(5 == len(net))
        #Bias node should always give an output of 1
        #bias = net.bias_node
        inputs = [1.0, 1.0]
        
        #Set all weights to 1
        for node in net.hidden_nodes:
            for key, val in node.weights.iteritems():
                node.weights[key] = 1.0
        for key, val in net.output_nodes[0].weights.iteritems():
            net.output_nodes[0].weights[key] = 1.0
        
        one = net.update(inputs)
        print("Weights are 1: {0}".format(one))
        #Assert each individually
        for node in net.hidden_nodes:
            out_val = node.output(inputs)
            print("Node output is: {0}".format(out_val))
            assert(3.0 == out_val)
        #Each hidden should have an output of 3
        #So net output should be 2*3 + 1 = 7
        assert(one == 7.0)

    def test6NetworkGDTrain(self):
        import numpy as np
        #Train on XOR
        indata = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
        outdata = np.array([[0],
                   [1],
                   [1],
                   [0]])
        
        net = build_feedforward(2, 4, 1)
        
        net.learn(indata, outdata, epochs=500)
        
        print('network training xor results:')
        for result, target in zip(net.sim(indata), outdata):
            print(result[0], target[0])
            assert(round(result[0]) == target[0])
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testMultiplication']
    unittest.main()
