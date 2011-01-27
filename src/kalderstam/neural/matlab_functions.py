from random import uniform
from random import gauss
from random import random
import numpy
from kalderstam.neural.network import network
import matplotlib.pyplot as plt
from math import sqrt
from numpy.core.numeric import dot
import math

def loadsyn1(n = 100):
    half = n/2
    
    P = numpy.zeros([2, n])
    for row in range(len(P)):
        for col in range(len(P[row, :])):
            if (col >= half):
                P[row, col] = -0.5 + uniform(-1, 1)
            else:
                P[row, col] = 0.5 + uniform(-1, 1)
    
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    return (P, T)

def loadsyn2(n = 100):
    half = n/2
    
    P = numpy.zeros([2, n])
    row = 0
    for col in range(len(P[row, :])):
        if (col >= half):
            P[row, col] = 10+2*uniform(-1, 1)
        else:
            P[row, col] = -10+2*uniform(-1, 1)
            
    row = 1
    for col in range(len(P[row, :])):
        P[row, col] = 10*uniform(-1, 1)
        
    #Rotate it to make it interesting
    #R = numpy.array([[sqrt(2), sqrt(2)], [sqrt(2), sqrt(2)]])
    #P = dot(R, P)
    
    #And normalize
    #P[0,:] = P[0,:]/numpy.std(P[0,:])
    #P[1,:] = P[1,:]/numpy.std(P[1,:])
    
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    return (P, T)

def loadsyn3(n = 100):
    half = n/2
    
    Rpos = 0.6
    Rneg = 1.3
    
    P = numpy.zeros([2, n])
    
    row = 0
    for col in range(len(P[row, :])):
        if (col < half):
            P[row, col] = Rpos*gauss(0, 1)*math.cos((2.0)*math.pi*random())
        else:
            P[row, col] = (random() + Rneg)*math.cos((2.0)*math.pi*random())
            
    row = 1
    for col in range(len(P[row, :])):
        if (col < half):
            P[row, col] = Rpos*gauss(0, 1)*math.sin((2.0)*math.pi*random())
        else:
            P[row, col] = (random() + Rneg)*math.sin((2.0)*math.pi*random())
    
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    return (P, T)

def plot2d2c(net, P, T, figure=1):
    if len(P) != 2:
        print 'Error: Input is not of dimension 2'
    else:
        plt.figure(figure)
        plt.title("Blue are correctly classified, while Red are incorrectly classified.")
        type1 = T[0][0]
        for num in range(0, len(T)):
            results = net.update([P[0, num], P[1, num]])
            mark = ''
            if (T[num][0] == type1):
                mark = 'o'
            else:
                mark = '+'
            
            if (T[num][0] == results[0]):
                plt.plot(P[0, num], P[1, num], 'b' + mark)
            else:
                plt.plot(P[0, num], P[1, num], 'r' + mark)
        boundary(net, P)
        plt.show()
        
def boundary(net, P):
    if len(P) != 2:
        print 'Error: Input is not of dimension 2'
    else:
        min_X1 = P[0, 0]
        max_X1 = P[0, 0]
        min_X2 = P[1, 0]
        max_X2 = P[1, 0]
        for num in range(0, len(P[0, :])):
            if (P[1, num] > max_X2):
                max_X2 = P[1, num]
            elif (P[1, num] < min_X2):
                min_X2 = P[1, num]
            if (P[0, num] > max_X1):
                max_X1 = P[0, num]
            elif (P[0, num] < min_X1):
                min_X1 = P[0, num]
        
        x1_inc = (max_X1 - min_X1)/100
        x2_inc = (max_X2 - min_X2)/100
        
        x1 = min_X1
        x2 = min_X2
        
        coords = [[],[]]
        while (x1 < max_X1):
            
            x2 = min_X2
            prev_val = net.update([x1, x2])
            while (x2 < max_X2):
                val = net.update([x1, x2])
                if (val != prev_val):
                    coords[0].append(x1)
                    coords[1].append(x2)
                prev_val = val
                x2 += x2_inc
            x1 += x1_inc
        
        plt.plot(coords[0], coords[1], 'g--')
    
if __name__ == '__main__':
    
    #Binary activation function
    def activation_function(x):
        if x > 0:
            return 1
        else:
            return 0
        
    #logsig(n) = 1 / (1 + exp(-n))
    #tansig(n) = 2/(1+exp(-2*n))-1 = tanh(n)
    #net = newff(P,T,nodes,{'tansig' 'logsig'},[method]);
        
    P, T = loadsyn3(100)
                
    net = network()
    net.build_feedforward(2, 5, 1, output_function = activation_function)
    
    net.traingd(P, T, 300, 0.1)
    
    plot2d2c(net, P, T)