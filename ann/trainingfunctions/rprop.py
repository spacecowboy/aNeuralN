# -*- coding: utf-8 -*-

from __future__ import division
from itertools import izip
from collections import defaultdict
from ann.errorfunctions import SSE
from time import time
import numpy as np

def train_rprop(net, inputarray, targetarray, epochs, show = 50):
    '''RPROP'''
    #SumSquareError
    error = SSE()
    #Maximum and minimum weight changes
    d_max, d_min = 50, 0.00001
    #How much to adjust updates
    d_pos, d_neg = 1.2, 0.5
    #Previous weight changes, initialized to 0.1
    prev_updates = defaultdict(lambda: defaultdict(lambda: 0.1))
    prev_node_derivatives = defaultdict(lambda: defaultdict(lambda: 1))

    def sign(x):
        if x > 0:
            return 1
        if x < 0:
            return - 1
        else:
            return 0

    #Train epoch times
    for e in xrange(epochs):
        if show is not None and (e == 0 or not (e + 1) % show):
            results = net.sim(inputarray)
            print("Error: {}".format(error(targetarray, results)))
        #Each epoch is one pass through the inputs
        #Keep all new weights in a dict
        node_derivatives = defaultdict(lambda: defaultdict(lambda: 0))

        for inputs, targets in zip(inputarray, targetarray):
            gradients = defaultdict(lambda: 0)
            iter_id = time()
            #First let all nodes calculate their outputs this round
            net.output(inputs, iter_id)
            #Using the same iter_id, we will use cached values

            #Initialize with gradient on output nodes
            for node, target in zip(net.output_nodes, targets):
                gradients[node] = error.derivative(target, node.output(inputs, iter_id))

            #Initialize zeros for hidden nodes

            #Iterate over the nodes and calculate their weight corrections
            #Will start on output node
            #for node in net:
            for node in net.output_nodes + net.hidden_nodes:
            #Calculate local error gradient
                gradients[node] *= node.output_derivative()

                #Propagate the error backwards and then update the weights
                for back_node, back_weight in node.weights.iteritems():
                    #First try to use it as an input node
                    try:
                        index = int(back_node)
                        #Weight update is easily calculated
                        node_derivatives[node][back_node] += (gradients[node] * inputs[index])
                    except TypeError:
                        #Not an input, then it's a node
                        gradients[back_node] += back_weight * gradients[node]
                        #Calculate and add the weight update
                        node_derivatives[node][back_node] += (gradients[node] * back_node.output(inputs, iter_id))

        #Now apply the weight updates!
        for node, derivatives in node_derivatives.iteritems():
            new_updates = defaultdict(lambda: 0)
            set_to_zeroes = []
            for other_node, deriv in derivatives.iteritems():
                prev_update = prev_updates[node][other_node]
                prev_deriv = prev_node_derivatives[node][other_node]
                if prev_deriv * deriv > 0:
                    upd = min(abs(prev_update) * d_pos, d_max)
                    new_updates[other_node] = upd * sign(deriv)
                    node.weights[other_node] += new_updates[other_node]
                elif prev_deriv * deriv < 0:
                    upd = max(abs(prev_update) * d_neg, d_min)
                    new_updates[other_node] = upd * sign(deriv)
                    set_to_zeroes.append((other_node, 0))
                    node.weights[other_node] -= prev_update
                else:
                    new_updates[other_node] = abs(prev_update) * sign(deriv)
                    node.weights[other_node] += new_updates[other_node]

            derivatives.update(set_to_zeroes)
            prev_updates[node] = new_updates
        prev_node_derivatives = node_derivatives