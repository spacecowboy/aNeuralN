'''
Created on Apr 11, 2011

@author: jonask
'''
import unittest
import numpy as np
from kalderstam.neural.error_functions.cox_error import get_risk_groups, \
    calc_sigma, derivative_sigma, shift, derivative_error, calc_beta, \
    get_beta_force, derivative_beta
from kalderstam.util.numpyhelp import indexOf

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

    def testCalc_beta(self):
        """Calculate beta for a predetermined optimal value"""
        #Check that it diverges if given a perfect ordering
        outputs = np.array([[i] for i in np.linspace(0, 3, 100)])
        timeslots = np.arange(100) #0-99
        risk_groups = get_risk_groups(timeslots)

        #print outputs
        #print timeslots
        diverged = False
        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError:
            #print("Diverged")
            diverged = True #It should diverge in this case
        assert(diverged)
        #Just change one value, and it should now no longer diverge
        outputs[38], outputs[97] = outputs[97], outputs[38]

        try:
            beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        except FloatingPointError:
            #print("Diverged, when it shouldn't")
            assert()

        #Now test that beta is actually a reasonable results
        #That means that F(Beta) = 0 (or very close to zero at least)
        F_result = 0
        for s in timeslots:
            F_result += outputs[s] - weighted_avg[s]
        #print(str(F_result))
        assert(round(F_result, 5) == 0)

    def testDerivativeSigma(self):
        outputs, timeslots = self.generateRandomTestData(100)
        sigma = calc_sigma(outputs)
        avg = outputs.sum() / len(outputs)
        #First calculate it manually, then compare with function
        for i in range(len(outputs)):
            output = outputs[i]
            ds = (output - avg) / (len(outputs) * sigma)
            assert(ds == derivative_sigma(sigma, i, outputs))

    def testDerivativeError(self):
        outputs, timeslots = self.generateRandomTestData(100)
        sigma = calc_sigma(outputs)
        risk_groups = get_risk_groups(timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        testDE = -np.exp(shift - beta * sigma) / (np.exp(shift - beta * sigma) + 1)

        assert(testDE == derivative_error(beta, sigma))

    def testDerivativeBeta(self):
        outputs, timeslots = self.generateRandomTestData(100)
        sigma = calc_sigma(outputs)
        risk_groups = get_risk_groups(timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        beta_force = get_beta_force(beta_risk, part_func, weighted_avg, outputs, timeslots, risk_groups)

        exp_value = np.exp(beta * outputs)
        exp_value_yi = exp_value * outputs
        exp_value_yi2 = exp_value_yi * outputs

        dFdB = -(exp_value_yi.sum() / exp_value.sum())**2 - exp_value_yi2.sum() / exp_value.sum()
        for i in range(len(outputs)):
            output = outputs[i, 0]
            dFdYi = 0
            for s in range(len(outputs)):
                delta = 0
                if i == s:
                    delta = 1
                if i in risk_groups[s]:
                    dWdYi = np.exp(beta * output) / part_func[s] * (1 + beta * output + beta * weighted_avg[s])
                else:
                    dWdYi = 0
                dFdYi += delta - dWdYi

            dBdYi = -dFdYi / dFdB
            method_value = derivative_beta(beta, part_func, weighted_avg, beta_force, i, outputs, timeslots)
            print(dBdYi, method_value)
            assert(dBdYi == method_value)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
