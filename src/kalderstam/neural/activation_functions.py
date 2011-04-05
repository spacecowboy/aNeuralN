#A few activation functions
import numpy
from numpy import exp

def get_function(name):
    if (name == str(logsig())):
        return logsig()
    elif (name == str(linear())):
        return linear()
    elif (name == str(tanh())):
        return tanh()

class logsig():
    def __str__(self):
        return 'logsig'
    def function(self, x):
        return 1 / (1 + exp(-x))
    def derivative(self, x):
        #return exp(x)/((exp(x) + 1)**2)
        y = self.function(x)
        return y * (1 - y)

class linear():
    def __init__(self, a = 1):
        self.a = a
    def __str__(self):
        return 'linear'
    def function(self, x):
        return self.a * x
    def derivative(self, x): #@UnusedVariable
        return self.a

class tanh():
    def __str__(self):
        return 'tanh'
    def function(self, x):
        return numpy.tanh(x)
    def derivative(self, x):
        #return (1/math.cosh(x))**2 #sech(x)^2
        y = self.function(x)
        return (1 - y) * (1 + y)
