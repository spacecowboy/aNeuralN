'''
Created on Apr 11, 2011

@author: jonask
'''
import unittest
import numpy as np
from kalderstam.neural.error_functions.cox_error import get_risk_groups, \
    calc_sigma, derivative_sigma, shift, derivative_error, calc_beta, \
    get_beta_force, derivative_beta, get_y_force
from kalderstam.util.numpyhelp import indexOf
from random import sample

class Test(unittest.TestCase):

    def generateRandomTestData(self, number):
        outputs = np.random.random((number, 1))
        timeslots = np.array([num for num in sample(range(len(outputs)), len(outputs))])

        return (outputs, timeslots)

    def generateFixedData(self, number, outputs):
        timeslots = np.arange(number)

        return (outputs, timeslots)

    def testGetRiskGroups(self):
        outputs, timeslots = self.generateRandomTestData(50)
        risk_groups = get_risk_groups(timeslots)
        print risk_groups
        for group, start_index in zip(risk_groups, range(len(timeslots))):
            for ri, ti in zip(group, timeslots[start_index:]):
                assert(ri == ti)

    def testPartFunc(self):
        outputs, timeslots = self.generateRandomTestData(100)
        risk_groups = get_risk_groups(timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        for z, risk_group in zip(part_func, risk_groups):
            testz = np.sum(np.exp(beta * outputs[risk_group]))
            print(z, testz)
            assert(z == testz)

    def testWeightedAverage(self):
        outputs, timeslots = self.generateRandomTestData(100)
        risk_groups = get_risk_groups(timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        for w, z, risk_group in zip(weighted_avg, part_func, risk_groups):
            testw = 1 / z * np.sum(np.exp(beta * outputs[risk_group]) * outputs[risk_group])
            print(w, testw)
            assert(round(w, 10) == round(testw, 10))


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

    def testYForce(self):
        outputs, timeslots = self.generateRandomTestData(100)
        risk_groups = get_risk_groups(timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        for output_index in range(len(outputs)):
            #Do the derivative for every Yi
            output = outputs[output_index, 0]
            test_yforce = 0
            for es, risk_group, z, w in zip(timeslots, risk_groups, part_func, weighted_avg):
                if es == output_index:
                    delta = 1
                else:
                    delta = 0
                if output_index in risk_group:
                    wpart = np.exp(beta * output) / z * (1 + beta * (output + w))
                else:
                    wpart = 0
                test_yforce += delta - wpart

            yforce = get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)

            print(test_yforce, yforce)
            assert(test_yforce == yforce)


    def testBetaForce(self):
        outputs, timeslots = self.generateRandomTestData(100)
        risk_groups = get_risk_groups(timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)

        testbeta_force = 0
        for risk_group, z, w in zip(risk_groups, part_func, weighted_avg):
            exp_value = np.exp(beta * outputs[risk_group])
            exp_value_yi = exp_value * outputs[risk_group]
            exp_value_yi2 = exp_value_yi * outputs[risk_group]

            testbeta_force += -(exp_value_yi.sum() / exp_value.sum())**2 - exp_value_yi2.sum() / exp_value.sum()

        testbeta_force *= -1

        betaforce = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

        print(testbeta_force, betaforce)
        assert(round(testbeta_force, 10) == round(betaforce, 10))

    def testDerivativeBeta(self):
        outputs, timeslots = self.generateRandomTestData(100)
        sigma = calc_sigma(outputs)
        risk_groups = get_risk_groups(timeslots)
        beta, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots, risk_groups)
        beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

        for output_index in range(len(outputs)):
            output = outputs[output_index, 0]
            y_force = get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)

            dBdYi = -y_force / beta_force
            method_value = derivative_beta(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)
            print(dBdYi, method_value)
            assert(dBdYi == method_value)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
