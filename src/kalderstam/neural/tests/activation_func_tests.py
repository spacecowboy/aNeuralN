'''
Created on Apr 11, 2011

@author: jonask
'''
import unittest


class Test(unittest.TestCase):


    def testLinear(self):
        from kalderstam.neural.activation_functions import linear
        a = -4
        l = linear(a)
        for x in [37.5, -0.1, -55.09, 0.5756]:
            assert(l.function(x) == a * x)
            assert(l.derivative(-x) == a)

    def testLogsig(self):
        from kalderstam.neural.activation_functions import logsig
        from math import exp
        l = logsig()
        for x in [37.5, -0.1, -55.09, 0.5756]:
            assert(l.function(x) == 1 / (1 + exp(-x)))
            assert(round(l.derivative(x), 6) == round(exp(x) / ((exp(x) + 1) ** 2), 6)) #Need to round off since two floats will be "equal" but can differ in the 30th decimal or something. 6 decimals is good enough

    def testTanh(self):
        from kalderstam.neural.activation_functions import tanh
        from math import exp
        import math as m
        t = tanh()
        for x in [37.5, -0.1, -55.09, 0.5756]:
            assert(t.function(x) == m.tanh(x))
            assert(round(t.derivative(x), 6) == round((1 / m.cosh(x)) ** 2, 6))

    def testStrings(self):
        from kalderstam.neural.activation_functions import get_function, logsig, tanh, linear
        assert(isinstance(get_function('logsig'), logsig))
        assert(isinstance(get_function('linear'), linear))
        assert(isinstance(get_function('tanh'), tanh))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
