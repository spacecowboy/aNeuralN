from numpy import log, exp, array
import logging
import numpy as np
#import kalderstam.util.graphlogger as glogger
#cython file
import ccox_error as ccox

logger = logging.getLogger('kalderstam.neural.error_functions')

shift = 4 #Also known as Delta, it's the handwaving variable.

def get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg):
    beta_force = 0
    for risk_group, z, w in zip(risk_groups, part_func, weighted_avg):
        beta_force += -1 / z * np.sum(np.exp(beta * outputs[risk_group]) * outputs[risk_group] ** 2) + w ** 2

    return beta_force

def get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups):
    output = outputs[output_index, 0]
    beta_out = np.exp(beta * output)
    y_force = 0
    for es, risk_group, z, w in zip(timeslots, risk_groups, part_func, weighted_avg):
        #glogger.debugPlot('Partition function', part_func[s], style = 'b+')
        #
        kronicker = 0
        if es == output_index:
            kronicker = 1
        if output_index in risk_group:
            dy_part = beta_out / z * (1 + beta * (output - w))
        else:
            dy_part = 0
        y_force += kronicker - dy_part
    return y_force

def derivative_error(beta, sigma):
    """dE/d(Beta*Sigma)"""
    exp_value = exp(shift - beta * sigma)
    de = -exp_value / (1 + exp_value)
    #glogger.debugPlot('Error derivative', de, style = 'r.')
    return de

def derivative_betasigma(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups):
    """Derivative of (Beta*Sigma) with respect to y(i)"""
    bs = ccox.derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups) * sigma + beta * derivative_sigma(sigma, output_index, outputs) #@UndefinedVariable
    #glogger.debugPlot('BetaSigma derivative', bs, style = 'g+')
    if np.isnan(bs) or np.isinf(bs):
        raise FloatingPointError('Derivative BetaSigma is Nan or Inf: ' + str(bs))
    return bs

def derivative_sigma(sigma, output_index, outputs):
    """Eq. 12, derivative of Sigma with respect to y(i)"""
    output = outputs[output_index]
    ds = (output - outputs.mean()) / (len(outputs) * sigma)
    #glogger.debugPlot('Sigma derivative', ds, style = 'b+')
    return ds

def derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups):
    """Eq. 14, derivative of Beta with respect to y(i)"""
    #glogger.debugPlot('Beta derivative', res, style = 'r+')

    y_force = get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)
    #beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

    logger.info('OI:' + str(output_index) + ' B:' + str(beta / abs(beta)) + ' BF:' + str(beta_force / abs(beta_force)) + ' YF' + str(y_force / abs(y_force)))
    return - y_force / beta_force

def get_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots):
    result = 0
    for time_index in range(len(timeslots)):
        s = timeslots[time_index]
        output = outputs[s, 0]
        risk_outputs = outputs[risk_groups[time_index], 0]
        try:
            beta_risk[time_index] = exp(beta * risk_outputs)
        except FloatingPointError as e:
            logger.error("In get_slope for calc_beta: \n if beta is 40 and risk_output is -23, we will get an underflow.\n Setting numpy.seterr(under = 'warn') or 'ignore', will do set it to zero in that case.")
            raise(e)

        part_func[time_index] = np.sum(beta_risk[time_index])
        weighted_avg[time_index] = np.sum(beta_risk[time_index] * risk_outputs) / part_func[time_index]
        if np.isnan(weighted_avg[time_index]):
            #When beta is small enough, part_func will be zero. This means weighted avg is something divided by zero. raise exception
            raise FloatingPointError('Weighted avg (in get_slope) encountered a division by zero. Beta must be really small, Beta = ' + str(beta))
        result += (output - weighted_avg[time_index])

    return result

def calc_beta(outputs, timeslots, risk_groups):
    """Find the likelihood maximizing Beta numerically."""
    beta = 0.1 #Start with something small
    distance = 7.0 #Fairly large interval, we actually want to cross the zero

    beta_risk = [None for i in range(len(timeslots))]
    part_func = np.zeros(len(timeslots))
    weighted_avg = np.zeros(len(timeslots))

    slope = ccox.get_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots) #@UndefinedVariable

    not_started = True

    logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))
    #we will get overflow errors when beta goes above 710, but even 200 is completely unreasonable and means that beta will diverge. in that case, QUIT
    while abs(beta) < 200 and (abs(slope) > 0 or not_started) and abs(distance) > 0.0001: #Want positive beta, Some small limit close to zero, fix to make sure we try more than one value, stop when step size is too small
        not_started = False
        prev_slope = slope
        beta += distance
        slope = ccox.get_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots) #@UndefinedVariable
        if slope * prev_slope < 0:
            #Different signs, we have passed the zero point, change directions and half the distance
            distance /= -2
        elif abs(slope) > abs(prev_slope):
            #If the new slope is bigger than the last, we are going in the wrong direction
            distance *= -1
        logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))

    if abs(beta) >= 200:
        raise FloatingPointError('Beta is diverging')
    logger.debug("Beta = " + str(beta))
    return beta, beta_risk, part_func, weighted_avg

def calc_sigma(outputs):
    """Standard deviation, just use numpy for it. need ALL results, from net.sim(inputs)"""
    sigma = outputs.std()
    logger.debug("Sigma = " + str(sigma))
    return sigma

def get_risk_groups(timeslots):
    risk_groups = []
    for i in range(len(timeslots)):
        risk_groups.append(timeslots[i:])
    return risk_groups

def total_error(beta, sigma):
    """E = ln(1 + exp(Delta - Beta*Sigma))."""
    return log(1 + exp(shift - beta * sigma))

def derivative(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    return derivative_error(beta, sigma) * derivative_betasigma(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)
