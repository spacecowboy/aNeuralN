
import numpy
from kalderstam.neural.network import network
import matplotlib.pyplot as plt
from math import sqrt
from numpy.core.numeric import dot
import math

def loadsyn1(n = 100):
    half = n/2
    
    P = numpy.zeros([2, n])
                
    # The positives
    P[0, :half] = 0.5 + numpy.random.randn(half)
    P[1, :half] = 0.5 + numpy.random.randn(half)
    
    # The negatives
    P[0, half:n] = -0.5 + numpy.random.randn(n-half)
    P[1, half:n] = -0.5 + numpy.random.randn(n-half)
        
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    return (P, T)

def loadsyn2(n = 100):
    half = n/2
    
    P = numpy.zeros([2, n])
    
    # The positives
    P[0, :half] = 10 + 2.0*numpy.random.randn(half)
    P[1, :half] = 10*numpy.random.randn(half)
    
    # The negatives
    P[0, half:n] = -10 + 2.0*numpy.random.randn(n-half)
    P[1, half:n] = 10*numpy.random.randn(n-half)
    
        
    #Rotate it to make it interesting
    R = numpy.array([sqrt(2), sqrt(2), -sqrt(2), sqrt(2)]).reshape([2, 2])
    P = dot(R, P)
    
    #And normalize
    P[0,:] = P[0,:]/numpy.std(P[0,:])
    P[1,:] = P[1,:]/numpy.std(P[1,:])
    
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    return (P, T)

def loadsyn3(n = 100):
    half = n/2
    
    Rpos = 0.6
    Rneg = 0.9
    
    P = numpy.zeros([2, n])
    
    # The positives
    tmpang = 2.0*math.pi*numpy.random.rand(half)
    tmpr = Rpos*numpy.random.randn(half)
    P[0,:half] = tmpr*numpy.cos(tmpang)
    P[1,:half] = tmpr*numpy.sin(tmpang)
    
    # The negatives
    tmpang = 2.0*math.pi*numpy.random.rand(n-half)
    tmpr = numpy.random.rand(n-half) + Rneg
    P[0, half:n] = tmpr*numpy.cos(tmpang)
    P[1, half:n] = tmpr*numpy.sin(tmpang)
    
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
        
    #net = newff(P,T,nodes,{'tansig' 'logsig'},[method]);
        
    P, T = loadsyn2(100)
                
    net = network()
    net.build_feedforward(2, 1, 1, output_function = activation_function)
    
    net.traingd(P, T, 300, 0.1)
    
    plot2d2c(net, P, T)