import numpy
from math import sqrt
from numpy.core.numeric import dot
import math
import logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None #This makes matplotlib optional

logger = logging.getLogger('kalderstam.neural.matlab')

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

def plot2d2c(net, P, T, figure = 1, cut = 0.5):
    if len(P[0]) != 2:
        logger.error('Input is not of dimension 2')
    elif plt:
        plt.figure(figure)
        plt.title("Blue are correctly classified, while Red are incorrectly classified.")
        for x, y, target in zip(P[:, 0], P[:, 1], T[:, 0]):
            results = net.update([x, y])
            mark = ''
            if (target - 1 < -cut):
                mark = 'o'
            else:
                mark = '+'

            color = ''
            if (target > cut and results[0] > cut or
                target <= cut and results[0] <= cut):
                color = 'b'
            else:
                color = 'r'
            plt.plot(x, y, color + mark)
        boundary(net, P, cut)

def boundary(net, P, cut = 0.5):
    if len(P[0]) != 2:
        logger.error('Error: Input is not of dimension 2')
    elif plt:
        min_X1 = P[0, 0]
        max_X1 = P[0, 0]
        min_X2 = P[0, 1]
        max_X2 = P[0, 1]
        for num in xrange(0, len(P)):
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
            if prev_val > cut:
                prev_val = 1
            else:
                prev_val = 0
            while (x2 < max_X2):
                val = net.update([x1, x2])
                if val > cut:
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
        logger.info("ROC area: " + str(area) + "%")
        if plt:
            plt.figure(figure)
            plt.xlabel("1-specificity")
            plt.ylabel("sensitivity")
            plt.axis([-1, 101, -1, 101])
            plt.plot(x, y, 'r+', x, y, 'b-')
            plt.title("ROC area: " + str(area) + "%\nBest cut at: " + str(best_cut))
            plt.plot([0, best_x], [100, best_y], 'g-')

        return area, best_cut

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
            area += left_y * (x - left_x)
            if left_y != y:
                #And a triangle, will be negative if y has gone down, so no worries
                area += (x - left_x) * (y - left_y) / 2
            left_y = y
            left_x = x

        #Calculate distance
        dist = math.sqrt(((100 - y) ** 2) + (x ** 2))
        if dist <= best_dist:
            best_dist = dist
            best_cut = (x, y, cut)
    if area / 100 > 100 or area / 100 < 0:
        print(zip(x_array, y_array, cuts))
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

def plot_network_weights(net, figure = None):
    if plt:
        fig = plt.figure(figure)
        max = None
        #Get a weight matrix for the network
        weights = []
        for node in (list(net.hidden_nodes) + net.output_nodes):
            nweights = []
            #First check input nodes
            for i in xrange(net.num_of_inputs):
                if i in node.weights:
                    if max is None:
                        max = node.weights[i]
                    if node.weights[i] > abs(max):
                        max = node.weights[i]
                    nweights.append(node.weights[i])
                else:
                    nweights.append(0)
            for lnode in ([net.bias_node] + list(net.hidden_nodes) + net.output_nodes):
                if lnode in node.weights:
                    if max is None:
                        max = node.weights[lnode]
                    if abs(node.weights[lnode]) > abs(max):
                        max = node.weights[lnode]
                    nweights.append(node.weights[lnode])
                else:
                    nweights.append(0)
            weights.append(nweights)

        weights = numpy.matrix(weights).T
        
        #Plot it
        hinton(weights)
        
        # Set ticks so we know what we are looking at
        plt.yticks(numpy.arange(len(net) + 1)+0.5,
                   ["Output {0}".format(x) for x in reversed(xrange(len(net.output_nodes)))] +
                   ["Hidden {0}".format(x) for x in reversed(xrange(len(net.hidden_nodes)))] +
                   ["Bias"] +
                   ["Input {0}".format(x) for x in reversed(xrange(net.num_of_inputs))])
        plt.xticks(numpy.arange(1 + len(net.hidden_nodes) + len(net.output_nodes))+0.5,
                   ["Hidden {0}".format(x) for x in xrange(len(net.hidden_nodes))] +
                   ["Output {0}".format(x) for x in xrange(len(net.output_nodes))])
                   
        for tick in fig.get_axes()[0].yaxis.get_major_ticks():
            #tick.label1On = False
            tick.label2On = True
        for tick in fig.get_axes()[0].xaxis.get_major_ticks():
            tick.label1On = False
            tick.label2On = True

        #Rotate the ticks on the X-axis                   
        plt.setp( plt.gca().get_xticklabels(), rotation=45, horizontalalignment='left')
        
        plt.ylabel("Weights")
        plt.xlabel("Nodes\nBiggest absolute weight = " + str(max))
                   
        #This might put the graph of center, limit the axes
        plt.ylim(0, len(net) + 1)
        plt.xlim(0, len(net.hidden_nodes) + len(net.output_nodes))
        #Say what the max value is
        #plt.title("Biggest absolute weight = " + str(max))
        
        #Make sure stuff don't overlap
        #plt.tight_layout()
        fig.subplots_adjust(bottom=0.04, top=0.93, left=0.0, right=1.0)

def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    if plt:
        hs = numpy.sqrt(area) / 2
        xcorners = numpy.array([x - hs, x + hs, x + hs, x - hs])
        ycorners = numpy.array([y - hs, y - hs, y + hs, y + hs])
        plt.fill(xcorners, ycorners, colour, edgecolor = colour)

def hinton(W, maxWeight = None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    """
    if plt:
        #plt.clf()
        height, width = W.shape
        if not maxWeight:
            maxWeight = 2 ** numpy.ceil(numpy.log(numpy.max(numpy.abs(W))) / numpy.log(2))

        plt.fill(numpy.array([0, width, width, 0]), numpy.array([0, 0, height, height]), 'gray')
        #plt.axis('off')
        plt.axis('scaled')
        for x in xrange(width):
            for y in xrange(height):
                _x = x + 1
                _y = y + 1
                w = W[y, x]
                if w > 0:
                    _blob(_x - 0.5, height - _y + 0.5, min(1, w / maxWeight), 'white')
                elif w < 0:
                    _blob(_x - 0.5, height - _y + 0.5, min(1, -w / maxWeight), 'black')

        #plt.show()
