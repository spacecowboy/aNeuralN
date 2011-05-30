from kalderstam.neural.fast_network import Node
#from kalderstam.neural.network import node as Node
from random import uniform
import numpy as np
from kalderstam.neural.network import connect_node, build_feedforward

n = Node(active = "logsig", random_range = 1)

print(n.__class__)
print(n.__class__.__mro__)

print(n)
print(n.weights)
print(n.bias)
print(n.random_range)
print(n.activation_function)
print(dir(n))

thelist = np.array([100.0,200.0,300.0,400.0])
print thelist

n2 = Node(active = "tanh", random_range=1)
connect_node(n, n2)
connect_node(n2, 2)

print("manual calc is: ", n.weights[n2]*(n2.weights[2]*thelist[2] + n2.bias)+n.bias)

print("output: ", n.output(thelist))
print("output deriv: ", n2.output_derivative(thelist))

    
n3 = Node(weights={n2:0.5})
print(dir(n3))
    
print("Pickle test")

class ptest:
    def __new__(self):
        print("New here!")
    def __init__(self):
        print("Init here!")
        self.var = "hej"

import pickle


p = ptest()
print pickle.dumps(p)
print("buh")
print(dir(Node.__new__))
print(type(n2.__class__))

# Pickle dictionary using protocol 0.
print(n.__reduce__())
print pickle.dumps(n, -1)
print("right")

# Pickle the list using the highest protocol available.
print(n.weights)
print pickle.dumps(n.weights, -1)

print("done")

s = pickle.dumps(n, -1)

n_again = pickle.loads(s)

print(n.bias, n_again.bias)
print(n.activation_function, n_again.activation_function)
print(n.weights, n_again.weights)
print(n.output(thelist), n_again.output(thelist))

net = build_feedforward(4, 4, 1)
print net.update(thelist)

print "pickling net"
ns = pickle.dumps(net, -1)
print ns
pnet = pickle.loads(ns)
print pnet
print pnet.update(thelist)
print "pickle net done"