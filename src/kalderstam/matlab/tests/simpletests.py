import logging
from ..matlab_functions import loadsyn1, loadsyn2, loadsyn3, plot2d2c, plot_network_weights, plotroc, stat

import unittest


class Test(unittest.TestCase):


    def testName(self):
        logging.basicConfig(level = logging.DEBUG)

        from kalderstam.neural.network import build_feedforward

        P, T = loadsyn1(100)
        P, T = loadsyn2(100)
        P, T = loadsyn3(100)

        net = build_feedforward(2, 3, 1)

        plot_network_weights(net, figure = 3)

        Y = net.sim(P)

        [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)

        plotroc(Y, T)
        plot2d2c(net, P, T, figure = 2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
