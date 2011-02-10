#A few activation functions
import math
from math import exp

class logsig():
    def function(self, x):
        return 1 / (1 + exp(-x))
    def derivative(self, x):
        #return exp(x)/((exp(x) + 1)**2)
        y = self.function(x)
        return y*(1 - y)

class linear():
    def function(self, x):
        return x
    def derivative(self, x):
        return 1

class tanh():
    def function(self, x):
        return math.tanh(x)
    def derivative(self, x):
        #return (1/math.cosh(x))**2 #sech(x)^2
        y = self.function(x)
        return (1 - y)*(1 + y)