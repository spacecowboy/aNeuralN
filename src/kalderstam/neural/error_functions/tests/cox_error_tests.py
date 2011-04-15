'''
Created on Apr 11, 2011

@author: jonask
'''
import unittest
import numpy as np
from ..cox_error import get_risk_outputs as pyget_risk_outputs, get_slope as pyget_slope, derivative_beta as pyderivative_beta, calc_beta
from ..ccox_error import get_risk_outputs as cget_risk_outputs, get_slope as cget_slope, derivative_beta as cderivative_beta #@UnresolvedImport
from kalderstam.util.numpyhelp import indexOf
from kalderstam.neural.error_functions.cox_error import get_risk_groups
from _xmlplus.xpath.XPathParser import SELF
from kalderstam.util.decorators import benchmark

class Test(unittest.TestCase):

    def generateRandomTestData(self, number):
        outputs = np.random.random((number, 1))
        return self.generateFixedData(number, outputs)

    def generateFixedData(self, number, outputs):
        timeslots = np.arange(number)
#        sorted_outputs = np.sort(outputs, axis = 0)
#        timeslots = np.zeros(number, dtype = int)
#        for i in range(len(timeslots)):
#            timeslots[i] = indexOf(outputs, sorted_outputs[i])[0]

        return (outputs, timeslots)

    def testGetRiskGroups(self):
        """For this data, the risk groups should just be 0,1,2,3. 1,2,3. 2,3. 3. etc"""
        outputs, timeslots = self.generateRandomTestData(50)
        risk_groups = get_risk_groups(timeslots)
        print risk_groups
        for group in risk_groups:
            prev = group[0]
            for i in group[1:]:
                if prev > i:
                    assert()
                else:
                    prev = i

    def testCythonDerivative_beta(self):
        """Make sure the cython code returns the same values as python code."""
        outputs, timeslots = self.generateRandomTestData(100)

        beta = 0.79 #Start with something small

        risk_outputs = [None for i in range(len(timeslots))]
        beta_risk = [None for i in range(len(timeslots))]
        part_func = np.zeros(len(timeslots))
        weighted_avg = np.zeros(len(timeslots))

        cget_slope(beta, risk_outputs, beta_risk, part_func, weighted_avg, outputs, timeslots)
        beta_force = sum([-(beta_risk[s] * risk_outputs[s] ** 2).sum() / part_func[s] + weighted_avg[s] ** 2 for s in timeslots])
        beta_force *= -1

        for output_index in range(len(outputs)):
            cder = cderivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots)
            pyder = pyderivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots)
            print(cder, pyder)
            assert(isinstance(pyder, cder.__class__))
            assert(round(cder, 8) == round(pyder, 8))

    def testCythonGet_slope(self):
        """Make sure the cython code returns the same values as python code."""
        outputs, timeslots = self.generateRandomTestData(100)

        beta = 0.79 #Start with something small

        crisk_outputs = [None for i in range(len(timeslots))]
        pybeta_risk = [None for i in range(len(timeslots))]
        cbeta_risk = [None for i in range(len(timeslots))]
        pypart_func = np.zeros(len(timeslots))
        cpart_func = np.zeros(len(timeslots))
        pyweighted_avg = np.zeros(len(timeslots))
        cweighted_avg = np.zeros(len(timeslots))

        risk_groups = get_risk_groups(timeslots)

        pyslope = pyget_slope(beta, risk_groups, pybeta_risk, pypart_func, pyweighted_avg, outputs, timeslots)
        cslope = cget_slope(beta, crisk_outputs, cbeta_risk, cpart_func, cweighted_avg, outputs, timeslots)

        #Check equality between all returned values
        #print(pyslope, pyslope.__class__)
        #print(cslope, cslope.__class__)
        assert(not np.isnan(pyslope))
        assert(not np.isnan(cslope))
        #assert(isinstance(pyslope, cslope.__class__)) #Make sure they are of the same type
        assert(pyslope == cslope)
        #Risk outputs
        for i, c in zip(range(len(crisk_outputs)), crisk_outputs):
            p = outputs[risk_groups[i], 0]
            assert(isinstance(p, c.__class__))
            print(p, p.__class__)
            print(c, c.__class__)
            for pp, cc in zip(p, c):
                print(pp, cc)
                assert(isinstance(pp, cc.__class__))
                assert(pp == cc)
        #Beta risk
        for p, c in zip(pybeta_risk, cbeta_risk):
            assert(isinstance(p, c.__class__))
            for pp, cc in zip(p, c):
                print(pp, cc)
                assert(isinstance(pp, cc.__class__))
                assert(pp == cc)
        #part_func
        for p, c in zip(pypart_func, cpart_func):
            print(p, c)
            assert(isinstance(p, c.__class__))
            assert(p == c)
        #weighted average
        for p, c in zip(pyweighted_avg, cweighted_avg):
            print(p, c)
            assert(isinstance(p, c.__class__))
            assert(p == c)


    def testCythonGet_risk_outputs(self):
        """Make sure the cython code returns the same values as python code."""

        outputs, timeslots = self.generateRandomTestData(100)
        risk_groups = get_risk_groups(timeslots)
        for time_index in range(len(timeslots)):
            risks = outputs[risk_groups[time_index]]
            py_risks = pyget_risk_outputs(time_index, timeslots, outputs)
            cy_risks = cget_risk_outputs(time_index, timeslots, outputs) #@UndefinedVariable
            assert(len(py_risks) == len(timeslots) - time_index)
            assert(len(cy_risks) == len(timeslots) - time_index)
            assert(len(risks) == len(py_risks))
            #Compare values in risk_groups
            print(cy_risks)
            for index in range(len(py_risks)):
                assert(py_risks[index] == cy_risks[index])
                assert(py_risks[index] == risks[index])

    def testCalc_beta(self):
        """Calculate beta for a predetermined optimal value"""
        #Check that it diverges if given a perfect ordering
        outputs = np.array([[i] for i in np.linspace(0, 3, 100)])
        timeslots = np.arange(100) #0-99
        risk_groups = get_risk_groups(timeslots)

        print outputs
        print timeslots
        diverged = False
        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError:
            print("Diverged")
            diverged = True #It should diverge in this case
        assert(diverged)
        #Just change one value, and it should now no longer diverge
        outputs[38], outputs[97] = outputs[97], outputs[38]

        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError:
            print("Diverged, when it shouldn't")
            assert()

        #Now test that beta is actually a reasonable results
        #That means that F(Beta) = 0 (or very close to zero at least)
        F_result = 0
        for s in timeslots:
            F_result += outputs[s] - weighted_avg[s]
        print(str(F_result))
        assert(round(F_result, 5) == 0)

    def testDerivativeSigma():


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
