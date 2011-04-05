from numpy import log, exp, array
import logging
import numpy as np
#import kalderstam.util.graphlogger as glogger

logger = logging.getLogger('kalderstam.neural.error_functions')

shift = 4 #Also known as Delta, it's the handwaving variable.
#sigma = None #Must be calculated before training begins
#beta = None #Must be calculated before training begins

def derivative_error(beta, sigma):
    """dE/d(Beta*Sigma)"""
    de = -(exp(shift - beta * sigma)) / (1 + exp(shift - beta * sigma))
    #glogger.debugPlot('Error derivative', de, style = 'r.')
    return de

def derivative_betasigma(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots):
    """Derivative of (Beta*Sigma) with respect to y(i)"""
    bs = derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots) * sigma + beta * derivative_sigma(sigma, output_index, outputs)
    #glogger.debugPlot('BetaSigma derivative', bs, style = 'g+')
    return bs

def derivative_sigma(sigma, output_index, outputs):
    """Eq. 12, derivative of Sigma with respect to y(i)"""
    output = outputs[output_index]
    ds = (output - outputs.mean()) / (len(outputs) * sigma)
    #glogger.debugPlot('Sigma derivative', ds, style = 'b+')
    return ds

def derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots):
    """Eq. 14, derivative of Beta with respect to y(i)"""
    output = outputs[output_index]
    y_force = 0
    beta_out = exp(beta * output)
    for s in timeslots:
        #glogger.debugPlot('Partition function', part_func[s], style = 'b+')
        kronicker = 0
        if s == output_index:
            kronicker = 1
        y_force += kronicker - beta_out / part_func[s] * (1 + beta * (output - weighted_avg[s]))

    res = -y_force / beta_force
    #glogger.debugPlot('Beta derivative', res, style = 'r+')
    return res

def get_slope(beta, risk_outputs, beta_risk, part_func, weighted_avg, outputs, timeslots):
    result = 0
    for s in timeslots:
        output = outputs[s]
        risk_outputs[s] = get_risk_outputs(s, timeslots, outputs)
        try:
            beta_risk[s] = exp(beta * risk_outputs[s])
        except FloatingPointError as e:
            logger.error("In get_slope for calc_beta: \n if beta is 40 and risk_output is -23, we will get an underflow.\n Setting numpy.seterr(under = 'warn') or 'ignore', will do set it to zero in that case.")
            raise(e)

        part_func[s] = beta_risk[s].sum()
        weighted_avg[s] = (beta_risk[s] * risk_outputs[s]).sum() / part_func[s]
        result += (output - weighted_avg[s])

    return result

def calc_beta(outputs, timeslots):
    """Find the likelihood maximizing Beta numerically."""
    beta = 20 #Start with something small
    distance = -7.0 #Fairly large interval, we actually want to cross the zero

    risk_outputs = [None for i in range(len(timeslots))]
    beta_risk = [None for i in range(len(timeslots))]
    part_func = np.zeros(len(timeslots))
    weighted_avg = np.zeros(len(timeslots))

    slope = get_slope(beta, risk_outputs, beta_risk, part_func, weighted_avg, outputs, timeslots)

    not_started = True

    logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))
    #we will get overflow errors when beta goes above 710, but even 200 is completely unreasonable and means that beta will diverge. in that case, QUIT
    while beta < 200 and (abs(slope) > 0 or not_started) and abs(distance) > 0.001: #Want positive beta, Some small limit close to zero, fix to make sure we try more than one value, stop when step size is too small
        not_started = False
        prev_slope = slope
        beta += distance
        slope = get_slope(beta, risk_outputs, beta_risk, part_func, weighted_avg, outputs, timeslots)
        if slope * prev_slope < 0:
            #Different signs, we have passed the zero point, change directions and half the distance
            distance /= -2
        logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))

    if beta >= 200:
        raise FloatingPointError('Beta is diverging')
    logger.info("Beta = " + str(beta))
    return beta, risk_outputs, beta_risk, part_func, weighted_avg

def calc_sigma(outputs):
    """Standard deviation, just use numpy for it. need ALL results, from net.sim(inputs)"""
    sigma = outputs.std()
    logger.info("Sigma = " + str(sigma))
    return sigma

def get_risk_outputs(s, timeslots, outputs):
    """s corresponds to the index of an output in outputs"""
    risk_outputs = []
    in_risk = False
    for index in timeslots:
        if s == index:
            in_risk = True #Will make sure that events that come after this are added to the risk group
        if in_risk:
            risk_outputs.append(outputs[index])
    return array(risk_outputs)

def total_error(beta, sigma):
    """E = ln(1 + exp(Delta - Beta*Sigma))."""
    return log(1 + exp(shift - beta * sigma))

def derivative(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    return derivative_error(beta, sigma) * derivative_betasigma(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots)
