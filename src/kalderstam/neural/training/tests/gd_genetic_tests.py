from ..committee import train_committee
from ..genetic import train_evolutionary
from ..gradientdescent import traingd
from kalderstam.neural.network import build_feedforward
from kalderstam.util.filehandling import get_validation_set
import logging
import unittest
from kalderstam.matlab.matlab_functions import stat, loadsyn3
from kalderstam.util.decorators import benchmark
from kalderstam.neural.network import build_feedforward_committee

logging.basicConfig(level = logging.DEBUG)

class Test(unittest.TestCase):


    def testSingleImprovement(self):
        P, T = loadsyn3(100)

        test, validation = get_validation_set(P, T)
        net1 = build_feedforward(2, 6, 1)

        epochs = 10

        P, T = test
        Y = net1.sim(P)
        [num_correct_first, num_correct_second, initial_test_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable

        P, T = validation
        Y = net1.sim(P)
        [num_correct_first, num_correct_second, initial_val_performance, num_first, num_second, missed] = stat(Y, T) #@UnusedVariable


        best1 = benchmark(train_evolutionary)(net1, test, validation, epochs, random_range = 5)

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

        best2 = benchmark(traingd)(net2, test, validation, epochs, block_size = 10, stop_error_value = 0)

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

        come = build_feedforward_committee(size = 4, input_number = 2, hidden_number = 6, output_number = 1)

        Y = come.sim(p)
        [num_correct_first, num_correct_second, initial_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        benchmark(train_committee)(come, train_evolutionary, p, t, epochs, random_range = 1)

        Y = come.sim(p)
        [num_correct_first, num_correct_second, genetic_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        print(initial_performance, genetic_performance)
        assert(initial_performance < genetic_performance)

        comg = build_feedforward_committee(size = 4, input_number = 2, hidden_number = 6, output_number = 1)
        epochs = 100
        Y = comg.sim(p)
        [num_correct_first, num_correct_second, initial_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        benchmark(train_committee)(comg, traingd, p, t, epochs, block_size = 10, stop_error_value = 0)

        Y = comg.sim(p)
        [num_correct_first, num_correct_second, gd_performance, num_first, num_second, missed] = stat(Y, t) #@UnusedVariable

        print(initial_performance, genetic_performance, gd_performance)
        assert(initial_performance < gd_performance)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
