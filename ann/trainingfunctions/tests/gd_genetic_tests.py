from ann.mp_network import train_committee
from ann.trainingfunctions.genetic import train_evolutionary
from ann.trainingfunctions.gradientdescent import traingd
from ann.errorfunctions import sumsquare_total
from ann.network import build_feedforward
from ann.filehandling import get_validation_set
import logging
import math
import numpy
import unittest
from ann.network import build_feedforward_committee

#logging.basicConfig(level = logging.DEBUG)

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

class Test(unittest.TestCase):


    def testSingleImprovement(self):
        P, T = loadsyn3(100)

        test, validation = get_validation_set(P, T)
        net1 = build_feedforward(2, 6, 1)

        epochs = 100

        P, T = test
        Y = net1.sim(P)
        [num_correct_first, num_correct_second, initial_test_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        P, T = validation
        Y = net1.sim(P)
        [num_correct_first, num_correct_second, initial_val_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable


        best1 = train_evolutionary(net1, test, validation, epochs, random_range = 5)

        P, T = test
        Y = best1.sim(P)
        [num_correct_first, num_correct_second, genetic_test_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        P, T = validation
        Y = best1.sim(P)
        [num_correct_first, num_correct_second, genetic_val_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        #Test sets
        print(initial_test_performance, genetic_test_performance)
        assert(initial_test_performance < genetic_test_performance)
        #Validation sets
        #print(initial_val_performance, genetic_val_performance)
        #assert(initial_val_performance < genetic_val_performance)

        net2 = build_feedforward(2, 6, 1)

        epochs = 100
        P, T = test
        Y = net2.sim(P)
        [num_correct_first, num_correct_second, initial_test_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        P, T = validation
        Y = net2.sim(P)
        [num_correct_first, num_correct_second, initial_val_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        best2 = traingd(net2, test, validation, epochs, block_size = 10)

        P, T = test
        Y = best2.sim(P)
        [num_correct_first, num_correct_second, gd_test_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        P, T = validation
        Y = best2.sim(P)
        [num_correct_first, num_correct_second, gd_val_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        #Assert that an improvement has occurred in each step
        #Test sets
        print(initial_test_performance, genetic_test_performance, gd_test_performance)
        assert(initial_test_performance < gd_test_performance)
        #Validation sets
        #print(initial_val_performance, genetic_val_performance, gd_val_performance)
        #assert(initial_val_performance < gd_val_performance)

    def testXCommitteeImprovement(self):
        P, T = loadsyn3(100)
        p, t = P, T

        epochs = 10

        come = build_feedforward_committee(size = 3, input_number = 2, hidden_number = 6, output_number = 1)

        Y = come.sim(p)
        [num_correct_first, num_correct_second, initial_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        train_committee(come, train_evolutionary, p, t, epochs = epochs, error_function = sumsquare_total)

        Y = come.sim(p)
        [num_correct_first, num_correct_second, genetic_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        print(initial_performance, genetic_performance)
        assert(initial_performance < genetic_performance)

        comg = build_feedforward_committee(size = 3, input_number = 2, hidden_number = 6, output_number = 1)
        epochs = 100
        Y = comg.sim(p)
        [num_correct_first, num_correct_second, initial_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        train_committee(comg, traingd, p, t, epochs = epochs, block_size = 10, error_function = sumsquare_total)

        Y = comg.sim(p)
        [num_correct_first, num_correct_second, gd_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        print(initial_performance, genetic_performance, gd_performance)
        assert(initial_performance < gd_performance)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
