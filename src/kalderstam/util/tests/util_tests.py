'''
Created on Apr 5, 2011

@author: jonask
'''
import unittest

class Test(unittest.TestCase):

    def testNumpyParseDataInputs(self):
        from ..filehandling import parse_data

        failed = False
        try:
            parse_data([[0, 1], [2, 3]])
        except TypeError as e:
            #We want an error here!
            failed = True
        assert(failed)

    def testNumpyHelp(self):
        from ..numpyhelp import indexOf
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
        from kalderstam.util.filehandling import (save_network, load_network,
        save_committee, load_committee, parse_file, get_validation_set,
        get_stratified_validation_set)
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

        print("All tests completed successfully!")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFilehandling']
    unittest.main()
