'''
Created on Apr 12, 2011

@author: jonask
'''
import unittest
from kalderstam.neural.network import build_feedforward


class Test(unittest.TestCase):


    def testMultiplication(self):
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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testMultiplication']
    unittest.main()
