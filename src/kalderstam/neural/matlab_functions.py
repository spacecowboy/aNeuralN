import numpy
import matplotlib.pyplot as plt
from math import sqrt
from numpy.core.numeric import dot
import math
import logging

logger = logging.getLogger('kalderstam.neural.matlab_functions')

def loadsyn1(n = 100):
    half = n / 2
    
    P = numpy.zeros([2, n])
                
    # The positives
    P[0, :half] = 0.5 + numpy.random.randn(half)
    P[1, :half] = 0.5 + numpy.random.randn(half)
    
    # The negatives
    P[0, half:n] = -0.5 + numpy.random.randn(n - half)
    P[1, half:n] = -0.5 + numpy.random.randn(n - half)
        
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    #Fix axes
    P = P.swapaxes(0, 1)
    
    return (P, T)

def loadsyn2(n = 100):
    half = n / 2
    
    P = numpy.zeros([2, n])
    
    # The positives
    P[0, :half] = 10 + 2.0 * numpy.random.randn(half)
    P[1, :half] = 10 * numpy.random.randn(half)
    
    # The negatives
    P[0, half:n] = -10 + 2.0 * numpy.random.randn(n - half)
    P[1, half:n] = 10 * numpy.random.randn(n - half)
    
        
    #Rotate it to make it interesting
    R = numpy.array([sqrt(2), sqrt(2), -sqrt(2), sqrt(2)]).reshape([2, 2])
    P = dot(R, P)
    
    #And normalize
    P[0, :] = P[0, :] / numpy.std(P[0, :])
    P[1, :] = P[1, :] / numpy.std(P[1, :])
    
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    #Fix axes
    P = P.swapaxes(0, 1)
    
    return (P, T)

def loadsyn3(n = 100):
    half = n / 2
    
    Rpos = 0.6
    Rneg = 0.9
    
    P = numpy.zeros([2, n])
    
    # The positives
    tmpang = 2.0 * math.pi * numpy.random.rand(half)
    tmpr = Rpos * numpy.random.randn(half)
    P[0, :half] = tmpr * numpy.cos(tmpang)
    P[1, :half] = tmpr * numpy.sin(tmpang)
    
    # The negatives
    tmpang = 2.0 * math.pi * numpy.random.rand(n - half)
    tmpr = numpy.random.rand(n - half) + Rneg
    P[0, half:n] = tmpr * numpy.cos(tmpang)
    P[1, half:n] = tmpr * numpy.sin(tmpang)
    
    T = numpy.ones([n, 1])
    T[half:n, 0] = 0
    
    #Fix axes
    P = P.swapaxes(0, 1)
    
    return (P, T)

def plot2d2c(net, P, T, figure = 1):
    if len(P[0]) != 2:
        logger.error('Input is not of dimension 2')
    else:
        plt.figure(figure)
        plt.title("Blue are correctly classified, while Red are incorrectly classified.")
        for x, y, target in zip(P[:, 0], P[:, 1], T[:, 0]):
            results = net.update([x, y])
            mark = ''
            if (target - 1 < -0.5):
                mark = 'o'
            else:
                mark = '+'
            
            color = ''
            if (target > 0.5 and results[0] > 0.5 or
                target <= 0.5 and results[0] <= 0.5):
                color = 'b'
            else:
                color = 'r'   
            plt.plot(x, y, color + mark)
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
        
        x1_inc = (max_X1 - min_X1) / 100
        x2_inc = (max_X2 - min_X2) / 100
        
        x1 = min_X1
        x2 = min_X2
        
        coords = [[], []]
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
            
def plotroc(Y, T, figure = 1):
    plt.figure(figure)
    Y = Y.flatten()
    T = T.flatten()
    
    if len(Y) != len(T):
        logger.error("Y(" + str(len(Y)) + ") and T(" + str(len(T)) + ") are not the same length!")
        raise TypeError
    else:
        #Sort them
        zipped = zip(Y, T)
        zipped.sort()
        Y, T = zip(*zipped)
        Y = numpy.array(Y)
        T = numpy.array(T)
        
        x = numpy.array([])
        y = numpy.array([])
        cuts = numpy.array([])
        for cut in Y:
            [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T, cut)
            
            x = numpy.append(x, 100 - num_correct_first)
            y = numpy.append(y, num_correct_second)
            cuts = numpy.append(cuts, cut)
            
        #plt.figure(2)
        plt.xlabel("1-specificity")
        plt.ylabel("sensitivity")
        plt.axis([-1, 101, -1, 101])
        plt.plot(x, y, 'r+', x, y, 'b-')

        area, (best_x, best_y, best_cut) = __calc_area__(x, y, cuts)
        logger.info("ROC area: " + str(area) + "%")
        plt.title("ROC area: " + str(area) + "%\nBest cut at: " + str(best_cut))
        
        plt.plot([0, best_x],[100, best_y],'g-')
        
        return area

def get_rocarea_and_best_cut(Y, T):
    Y = Y.flatten()
    T = T.flatten()
    
    if len(Y) != len(T):
        logger.error("Y(" + str(len(Y)) + ") and T(" + str(len(T)) + ") are not the same length!")
        raise TypeError
    else:
        #Sort them
        zipped = zip(Y, T)
        zipped.sort()
        Y, T = zip(*zipped)
        Y = numpy.array(Y)
        T = numpy.array(T)
        
        x = numpy.array([])
        y = numpy.array([])
        cuts = numpy.array([])
        for cut in Y:
            [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T, cut)
            
            x = numpy.append(x, 100 - num_correct_first)
            y = numpy.append(y, num_correct_second)
            cuts = numpy.append(cuts, cut)

        area, (best_x, best_y, best_cut) = __calc_area__(x, y, cuts)
        
        return area, best_cut
    
def __calc_area__(x_array, y_array, cuts):
    """Find the largest value of Y where X is the same. """
    left_y = 0 #Previous height at the right side
    left_x = 0 #Right side of that area
    area = 0
    best_cut = None
    best_dist = 10000 #Diagonal of entire graph
    for x, y, cut in zip(x_array, y_array, cuts):
        if x == left_x:
            left_y = y
        elif x > left_x:
            #Simple square
            area += left_y*(x - left_x)
            if left_y <> y:
                #And a triangle, will be negative if y has gone down, so no worries
                area += (x - left_x)*(y-left_y)/2
            left_y = y
            left_x = x
            
        #Calculate distance
        dist = math.sqrt(((100-y)**2) + (x**2))
        if dist <= best_dist:
            best_dist = dist
            best_cut = (x, y, cut)
    if area/100 > 100 or area/100 < 0:
        print zip(x_array, y_array, cuts)
    return (area / 100, best_cut)
        
def stat(Y, T, cut = 0.5):
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
        raise TypeError
    else:
        num_second = len(T.compress((T < 0.5).flat)) #T is 1 or 0
        num_first = len(T.compress((T > 0.5).flat))
        
        num_correct_firsterr = len(T.compress(((T - Y) >= (1 - cut)).flat))
        num_correct_first = 100.0 * (num_first - num_correct_firsterr) / max(1, num_first)
        
        num_correct_seconderr = len(T.compress(((T - Y) < -cut).flat))
        num_correct_second = 100.0 * (num_second - num_correct_seconderr) / max(1, num_second)
        
        missed = num_correct_firsterr + num_correct_seconderr
        total_performance = 100.0 * (len(T) - missed) / len(T)
        
        #print("\nResults for the training:\n")
        #print("Total number of data: " + str(len(T)) + " (" + str(num_second) + " ones and " + str(num_first) + " zeros)")
        #print("Number of misses: " + str(missed) + " (" + str(total_performance) + "% performance)")
        #print("Specificity: " + str(num_correct_first) + "% (Success for class 0)")
        #print("Sensitivity: " + str(num_correct_second) + "% (Success for class 1)")
        
        return [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed]
    
def plot_network_weights(net, figure=1):
    plt.figure(figure)
    
    #Get a weight matrix for the network
    weights = []
    for node in net.get_all_nodes():
        nweights = []
        #First check input nodes
        for i in range(net.num_of_inputs):
            if i in node.weights:
                nweights.append(node.weights[i])
            else:
                nweights.append(0)
        for lnode in net.get_all_nodes():
            if lnode == node:
                nweights.append(node.bias)
            elif lnode in node.weights:
                nweights.append(node.weights[lnode])
            else:
                nweights.append(0)
        weights.append(nweights)
    
    weights = numpy.matrix(weights).T
    #Plot it
    hinton(weights)

def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = numpy.sqrt(area) / 2
    xcorners = numpy.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = numpy.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    """
    #plt.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**numpy.ceil(numpy.log(numpy.max(numpy.abs(W)))/numpy.log(2))

    plt.fill(numpy.array([0,width,width,0]),numpy.array([0,0,height,height]),'gray')
    plt.axis('off')
    plt.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    
    #plt.show()

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    
    from kalderstam.neural.training_functions import traingd_block
    from kalderstam.neural.network import build_feedforward
    from kalderstam.util.filehandling import get_validation_set
    
    #Binary activation function
    def activation_function(x):
        if x > 0:
            return 1
        else:
            return 0
        
    P, T = loadsyn1(100)
    test, validation = get_validation_set(P, T, validation_size = 0)
    P, T = test
                
    net = build_feedforward(2, 3, 1)
    
    plot_network_weights(net, figure=3)
    plt.title('Before training')
    
    net = traingd_block(net, test, validation, 100, block_size = 10, stop_error_value = False)
    
    Y = net.sim(P)
    
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    
    plotroc(Y, T)
    plot2d2c(net, P, T, figure = 2)
    
    plot_network_weights(net, figure=4)
    plt.title('After training')
    
    plt.show()
