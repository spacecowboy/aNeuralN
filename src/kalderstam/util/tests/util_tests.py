'''
Created on Apr 5, 2011

@author: jonask
'''
import unittest
from kalderstam.util.filehandling import get_cross_validation_sets

class Test(unittest.TestCase):

    def testNumpyParseDataInputs(self):
        from kalderstam.util.filehandling import parse_data

        failed = False
        try:
            parse_data([[0, 1], [2, 3]])
        except TypeError as e:
            #We want an error here!
            failed = True
        assert(failed)

    def testNumpyHelp(self):
        from kalderstam.util.numpyhelp import indexOf
        import numpy as np
        outputs = np.random.random((100))
        idx = (56)
        val = indexOf(outputs, outputs[idx])
        assert(val == idx)
        outputs = np.random.random((100, 1))
        idx = (56, 0)
        val = indexOf(outputs, outputs[idx])
        assert(val == idx)
        outputs = np.random.random((10, 5, 5))
        idx = (7, 3, 2)
        val = indexOf(outputs, outputs[idx])
        assert(val == idx)

    def testFilehandling(self):
        print("Testing network saving/loading")
        from kalderstam.neural.network import build_feedforward, build_feedforward_committee
        from os import path
        from kalderstam.util.filehandling import (save_network, load_network, save_committee, load_committee, parse_file, get_validation_set)
        net = build_feedforward()

        results1 = net.update([1, 2])

        print(results1)

        filename = path.join(path.expanduser("~"), "test.ann")
        print("saving and reloading")
        save_network(net, filename)

        net = load_network(filename)
        results2 = net.update([1, 2])
        print(results2)

        assert(abs(results1[0] - results2[0]) < 0.0001) #float doesn't handle absolutes so well
        print("Good, now testing committee...")

        com = build_feedforward_committee()
        results1 = com.update([1, 2])
        print(results1)

        filename = path.join(path.expanduser("~"), "test.anncom")
        print("saving and reloading")

        save_committee(com, filename)

        com = load_committee(filename)
        results2 = com.update([1, 2])
        print(results2)

        assert(abs(results1[0] - results2[0]) < 0.0001) #float doesn't handle absolutes so well

        print("Results are good. Testing input parsing....")
        filename = path.join(path.expanduser("~"), "ann_input_data_test_file.txt")
        print("First, split the file into a test set(80%) and validation set(20%)...")
        inputs, targets = parse_file(filename, targetcols = 5, ignorecols = [0, 1, 4], ignorerows = [])
        test, validation = get_validation_set(inputs, targets, validation_size = 0.5)
        print(len(test[0]))
        print(len(test[1]))
        print(len(validation[0]))
        print(len(validation[1]))
        assert(len(test) == 2)
        assert(len(test[0]) > 0)
        assert(len(test[1]) > 0)
        assert(len(validation) == 2)
        assert(len(validation[0]) > 0)
        assert(len(validation[1]) > 0)
        print("Went well, now expecting a zero size validation set...")
        test, validation = get_validation_set(inputs, targets, validation_size = 0)
        print(len(test[0]))
        print(len(test[1]))
        print(len(validation[0]))
        print(len(validation[1]))
        assert(len(test) == 2)
        assert(len(test[0]) > 0)
        assert(len(test[1]) > 0)
        assert(len(validation) == 2)
        assert(len(validation[0]) == 0)
        assert(len(validation[1]) == 0)
        print("As expected. Now a 100% validation set...")
        test, validation = get_validation_set(inputs, targets, validation_size = 1)
        print(len(test[0]))
        print(len(test[1]))
        print(len(validation[0]))
        print(len(validation[1]))
        assert(len(test) == 2)
        assert(len(test[0]) == 0)
        assert(len(test[1]) == 0)
        assert(len(validation) == 2)
        assert(len(validation[0]) > 0)
        assert(len(validation[1]) > 0)
        print("Now we test a stratified set...")
        test, validation = get_validation_set(inputs, targets, validation_size = 0.5, binary_column = 0)
        print(len(test[0]))
        print(len(test[1]))
        print(len(validation[0]))
        print(len(validation[1]))
        assert(len(test) == 2)
        assert(len(test[0]) > 0)
        assert(len(test[1]) > 0)
        assert(len(validation) == 2)
        assert(len(validation[0]) > 0)
        assert(len(validation[1]) > 0)
        print("Test with no targets, the no inputs")
        inputs, targets = parse_file(filename, ignorecols = [0, 1, 4], ignorerows = [])
        assert((targets.size) == 0)
        assert((inputs.size) > 0)
        inputs, targets = parse_file(filename, targetcols = 3, ignorecols = [0, 1, 2, 4, 5, 6, 7 , 8, 9], ignorerows = [])
        assert((targets.size) > 0)
        assert((inputs.size) == 0)

    def testCrossValidationSplitting(self):
        print("Testing crossvalidation splitting")
        import numpy
        inputs = numpy.array([[x] for x in xrange(100)])
        targets = inputs + 0.5
        data_sets = get_cross_validation_sets(inputs, targets, 9)

        for (ti, tt), (vi, vt) in data_sets:
            #Verify that validation does not occur in test and vice versa
            for i, t in zip(ti, tt):
                assert(i[0] not in vi)
                assert(t[0] not in vt)

        for set in xrange(len(data_sets)):
            (ti1, tt1), (vi1, vt1) = data_sets[set]
            for set2 in xrange(len(data_sets)):
                if set == set2:
                    continue
                (ti2, tt2), (vi2, vt2) = data_sets[set2]
                #Verify that validation does occur in test set and so on
                for i, t in zip(vi1, vt1):
                    assert(i[0] not in vi2)
                    assert(t[0] not in vt2)

        #shuffling test, make sure we don't scramble the data
        print("Testing cross shuffling")
        inputs = numpy.zeros((1000, 2), dtype = numpy.float64)
        inputs[:, 0] = numpy.linspace(100, 400, 1000)
        inputs[:, 1] = numpy.linspace(0, 200, 1000)

        def formula1(a, b):
            return a * b
        def formula2(a, b):
            return a + b

        targets = numpy.zeros((1000, 2), dtype = numpy.float64)
        targets[:, 0] = formula1(inputs[:, 0], inputs[:, 1])
        targets[:, 1] = formula2(inputs[:, 0], inputs[:, 1])

        data_sets = get_cross_validation_sets(inputs, targets, 4)

        for (ti, tt), (vi, vt) in data_sets:
            #Verify length
            assert(len(ti) == 3.0 / 4.0 * len(targets))
            assert(len(tt) == 3.0 / 4.0 * len(targets))
            assert(len(vi) == 1.0 / 4.0 * len(targets))
            assert(len(vt) == 1.0 / 4.0 * len(targets))
            #Verify that formulas are still correct
            for i, t in zip(ti, tt):
                assert(t[0] == formula1(i[0], i[1]))
                assert(t[1] == formula2(i[0], i[1]))
            for i, t in zip(vi, vt):
                assert(t[0] == formula1(i[0], i[1]))
                assert(t[1] == formula2(i[0], i[1]))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFilehandling']
    unittest.main()
