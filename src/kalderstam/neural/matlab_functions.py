
import numpy
import matplotlib.pyplot as plt
from math import sqrt
from numpy.core.numeric import dot
import math
import logging

logger = logging.getLogger('kalderstam.neural.matlab_functions')

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
    
    #Fix axes
    P = P.swapaxes(0, 1)
    
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
    
    #Fix axes
    P = P.swapaxes(0, 1)
    
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
    
    #Fix axes
    P = P.swapaxes(0, 1)
    
    return (P, T)

def plot2d2c(net, P, T, figure=1):
    if len(P[0]) != 2:
        logger.error('Input is not of dimension 2')
    else:
        plt.figure(figure)
        plt.title("Blue are correctly classified, while Red are incorrectly classified.")
        for num in range(0, len(T)):
            results = net.update([P[num, 0], P[num, 1]])
            mark = ''
            if (T[num][0] - 1 < -0.5):
                mark = 'o'
            else:
                mark = '+'
            
            color = ''
            if (T[num][0] > 0.5 and results[0] > 0.5 or
                T[num][0] <= 0.5 and results[0] <= 0.5):
                color = 'b'
            else:
                color = 'r'   
            plt.plot(P[num, 0], P[num, 1], color + mark)
        boundary(net, P)
        
def boundary(net, P):
    if len(P[0]) != 2:
        logger.error('Error: Input is not of dimension 2')
    else:
        min_X1 = P[0, 0]
        max_X1 = P[0, 0]
        min_X2 = P[0, 1]
        max_X2 = P[0, 1]
        for num in range(0, len(P)):
            if (P[num, 1] > max_X2):
                max_X2 = P[num, 1]
            elif (P[num, 1] < min_X2):
                min_X2 = P[num, 1]
            if (P[num, 0] > max_X1):
                max_X1 = P[num, 0]
            elif (P[num, 0] < min_X1):
                min_X1 = P[num, 0]
        
        x1_inc = (max_X1 - min_X1)/100
        x2_inc = (max_X2 - min_X2)/100
        
        x1 = min_X1
        x2 = min_X2
        
        coords = [[],[]]
        while (x1 < max_X1):
            
            x2 = min_X2
            prev_val = net.update([x1, x2])
            if prev_val > 0.5:
                prev_val = 1
            else:
                prev_val = 0
            while (x2 < max_X2):
                val = net.update([x1, x2])
                if val > 0.5:
                    val = 1
                else:
                    val = 0
                if (val != prev_val):
                    coords[0].append(x1)
                    coords[1].append(x2)
                prev_val = val
                x2 += x2_inc
            x1 += x1_inc
        
        plt.plot(coords[0], coords[1], 'g.')
        
def plotroc(Y, T, points = 100):
    
    Y = Y.flatten()
    T = T.flatten()
    
    if len(Y) != len(T):
        logger.error("Y(" + str(len(Y)) + ") and T(" + str(len(T)) + ") are not the same length!")
    else:
        x = numpy.array([])
        y = numpy.array([])
        for cut in numpy.linspace(0, 1, points):
            num_first = max(0,len(T.compress((T>cut).flat)))
            num_second = max(0,len(T.compress((T<cut).flat)))
            
            num_correct_firsterr = len(T.compress(((T-Y)<-cut).flat))
            num_correct_first = 0
            if num_first > 0:
                num_correct_first = 100.0*(num_first-num_correct_firsterr)/num_first
            
            num_correct_seconderr = len(T.compress(((T-Y)>cut).flat))
            num_correct_second = 0
            if num_second > 0:
                num_correct_second = 100.0*(num_second-num_correct_seconderr)/num_second
            
            x = numpy.append(x, 100 - num_correct_first)
            y = numpy.append(y, num_correct_second)
        plt.figure(2)
        plt.title("ROC curve")
        plt.axis([101, -1, -1, 101])
        plt.plot(x, y, 'r-', x, y, 'ro')
            
        
def stat(Y, T):
    """ Calculates the results for a single output classification
     problem. Y is the network output and T is the target output.

     The results are returned as
     num_correct_first = number of class 0 targets that were correctly classified
     num_correct_second = number of class 1 targets that were correctly classified
     tot = total performance
     None = number of class 1 in T
     Nzero = number of class 0 in T
     miss = number of missclassified targets"""
    Y = Y.flatten()
    T = T.flatten()
    
    if len(Y) != len(T):
        logger.error("Y(" + str(len(Y)) + ") and T(" + str(len(T)) + ") are not the same length!")
    else:
        num_second = max(1,len(T.compress((T<0.5).flat)))
        num_first = max(1,len(T.compress((T>0.5).flat)))
        
        num_correct_firsterr = len(T.compress(((T-Y)<-0.5).flat))
        num_correct_first = 100.0*(num_first-num_correct_firsterr)/num_first
        
        num_correct_seconderr = len(T.compress(((T-Y)>0.5).flat))
        num_correct_second = 100.0*(num_second-num_correct_seconderr)/num_second
        
        missed = sum(abs(numpy.round(Y)-T))
        total_performance = 100.0*(len(T)-missed)/len(T)
        
        print("\nResults for the training:\n")
        print("Total number of data: " + str(len(T)) + " (" + str(num_second) + " ones and " + str(num_first) + " zeros)")
        print("Number of misses: " + str(missed) + " (" + str(total_performance) + "% performance)")
        print("Specificity: " + str(num_correct_first) + "% (Success for class 0)")
        print("Sensitivity: " + str(num_correct_second) + "% (Success for class 1)")
        
        return [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed]
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    from kalderstam.neural.training_functions import traingd
    from kalderstam.neural.network import build_feedforward
    
    #Binary activation function
    def activation_function(x):
        if x > 0:
            return 1
        else:
            return 0
        
    P, T = loadsyn1(100)
                
    net = build_feedforward(2, 1, 1)
    
    net = traingd(net, P, T, 300, 0.1)
    
    Y = net.sim(P)
    
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    
    plotroc(Y, T)
    plot2d2c(net, P, T)
    plt.show()