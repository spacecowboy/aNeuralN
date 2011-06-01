'''
Created on Apr 18, 2011

@author: jonask
'''
import unittest
from kalderstam.neural.error_functions.cox_error import get_risk_groups, \
    get_beta_force, generate_timeslots, get_slope as pyget_slope, derivative_beta as pyderivative_beta, calc_beta, get_y_force as pygetyforce
#from ..cox_error import get_slope as pyget_slope, derivative_beta as pyderivative_beta, calc_beta, get_y_force as pygetyforce
from kalderstam.neural.error_functions.cox_error_in_c import derivative_beta as cderivative_beta, get_slope as cget_slope#, #get_y_force as cgetyforce #@UnresolvedImport
import numpy as np


class Test(unittest.TestCase):

    def generateRandomTestData(self, number):
        outputs = np.random.random((number, 2))
        for i in range(len(outputs)):
            outputs[i, 1] = np.random.randint(0, 2) #inclusive, exclusive
        timeslots = generate_timeslots(outputs)

        return (outputs, timeslots)

    def testCDerivative_beta(self):
        """Make sure the cython code returns the same values as python code."""
        beta = 0.79 #Start with something small

        outputs, timeslots = self.generateRandomTestData(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = self.generateRandomTestData(100)

        risk_outputs = [None for i in range(len(timeslots))]
        beta_risk = [np.zeros(len(risk_groups[i])) for i in range(len(risk_groups))]
        part_func = np.zeros(len(timeslots))
        weighted_avg = np.zeros(len(timeslots))

        pyget_slope(beta, risk_outputs, beta_risk, part_func, weighted_avg, outputs, timeslots)
        beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

        for output_index in range(len(outputs)):
            cder = cderivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)
            pyder = pyderivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)
            print(cder, pyder)
            assert(isinstance(pyder, cder.__class__))
            assert(round(cder, 20) == round(pyder, 20))
            assert(cder == pyder)

    def testCGet_slope(self):
        """Make sure the cython code returns the same values as python code."""
        #outputs, timeslots = self.generateRandomTestData(100)

        beta = 0.79 #Start with something small

        #risk_groups = get_risk_groups(timeslots)
        outputs, timeslots = self.generateRandomTestData(100)
        risk_groups = get_risk_groups(outputs, timeslots)
        #Now get a new set, that won't match this, so beta doesn't diverge
        outputs, rnd_timeslots = self.generateRandomTestData(100)

        pybeta_risk = [np.zeros(len(risk_groups[i])) for i in range(len(risk_groups))]
        cbeta_risk = [np.zeros(len(risk_groups[i])) for i in range(len(risk_groups))]
        pypart_func = np.zeros(len(timeslots))
        cpart_func = np.zeros(len(timeslots))
        pyweighted_avg = np.zeros(len(timeslots))
        cweighted_avg = np.zeros(len(timeslots))

        pyslope = pyget_slope(beta, risk_groups, pybeta_risk, pypart_func, pyweighted_avg, outputs, timeslots)
        cslope = cget_slope(beta, risk_groups, cbeta_risk, cpart_func, cweighted_avg, outputs, timeslots)

        #Check equality between all returned values
        #print(pyslope, pyslope.__class__)
        #print(cslope, cslope.__class__)
        assert(not np.isnan(pyslope))
        assert(not np.isnan(cslope))
        #assert(isinstance(pyslope, cslope.__class__)) #Make sure they are of the same type
        print(pyslope, cslope)
        assert(round(pyslope, 20) == round(cslope, 20))
        assert(pyslope == cslope)
#        #Risk outputs
#        for i, c in zip(range(len(crisk_outputs)), crisk_outputs):
#            p = outputs[risk_groups[i], 0]
#            assert(isinstance(p, c.__class__))
#            #print(p, p.__class__)
#            #print(c, c.__class__)
#            for pp, cc in zip(p, c):
#                #print(pp, cc)
#                assert(isinstance(pp, cc.__class__))
#                assert(pp == cc)
        #Beta risk
        print("Beta Risk")
        for p, c, in zip(pybeta_risk, cbeta_risk):
            assert(isinstance(p, c.__class__))
            for pp, cc in zip(p, c):
                print(pp, cc)
                assert(isinstance(pp, cc.__class__))
                #if (pp != cc):
                assert(pp == cc)
        #part_func
        print("Part func")
        for p, c in zip(pypart_func, cpart_func):
            print(p, c)
            assert(isinstance(p, c.__class__))
            assert(p == c)
        #weighted average
        print("Weighted avg")
        for p, c in zip(pyweighted_avg, cweighted_avg):
            print(p, c)
            assert(isinstance(p, c.__class__))
            assert(p == c)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
