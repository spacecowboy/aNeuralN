from numpy import log, exp, array
import logging
import numpy as np
#import kalderstam.util.graphlogger as glogger
from cox_error_in_c import derivative_beta as cderivative_beta, get_slope as cget_slope

logger = logging.getLogger('kalderstam.neural.error_functions')

shift = 4 #Also known as Delta, it's the handwaving variable.

def generate_timeslots(T):
    timeslots = np.array([], dtype = int)
    for x_index in range(len(T)):
        time = T[x_index][0]
        if len(timeslots) == 0:
            timeslots = np.insert(timeslots, 0, x_index)
        else:
            added = False
            #Find slot
            for index in range(len(timeslots)):
                time_index = timeslots[index]
                if time > T[time_index, 0]:
                    timeslots = np.insert(timeslots, index, x_index)
                    added = True
                    break
            if not added:
                #Reached the end, insert here
                timeslots = np.append(timeslots, x_index)

    return timeslots

def plot_correctly_ordered(outputs, timeslots):
    timeslots_network = generate_timeslots(outputs)
    global prev_timeslots_network
    if prev_timeslots_network is None:
        prev_timeslots_network = timeslots_network
    #Now count number of correctly ordered indices
    count = 0
    diff = 0
    for i, j, prev in zip(timeslots, timeslots_network, prev_timeslots_network):
        if i == j:
            count += 1
        if j != prev:
            diff += 1

    glogger.debugPlot('Network ordering difference', y = diff, style = 'r-')
    logger.info('Network ordering difference: ' + str(diff))
    prev_timeslots_network = timeslots_network

    countreversed = 0
    for i, j in zip(timeslots[::-1], timeslots_network):
        if i == j:
            countreversed += 1
    correct = max(count, countreversed)
    #glogger.debugPlot('Number of correctly ordered outputs', y = correct, style = 'r-')
    #logger.info('Number of correctly ordered outputs: ' + str(correct))

def get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg):
    beta_force = 0
    for risk_group, z, w in zip(risk_groups, part_func, weighted_avg):
        beta_force += -1 / z * np.sum(np.exp(beta * outputs[risk_group]) * outputs[risk_group] ** 2) + w ** 2

    return beta_force

def get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups):
    output = outputs[output_index, 0]
    #print beta, output
    beta_out = exp(beta * output)
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
    bs = derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups) * sigma + beta * derivative_sigma(sigma, output_index, outputs) #@UndefinedVariable
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
    return cderivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups)

    #y_force = get_y_force(beta, part_func, weighted_avg, output_index, outputs, timeslots, risk_groups)
    #beta_force = get_beta_force(beta, outputs, risk_groups, part_func, weighted_avg)

    #logger.info('OI:' + str(output_index) + ' B:' + str(beta / abs(beta)) + ' BF:' + str(beta_force / abs(beta_force)) + ' YF' + str(y_force / abs(y_force)))
    #return - y_force / beta_force

def get_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots):
    #cget_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots)
    result = 0
    for time_index in range(len(timeslots)):
        s = timeslots[time_index]
        output = outputs[s, 0]
        risk_outputs = outputs[risk_groups[time_index], 0]
        try:
            beta_risk[time_index] = np.exp(beta * risk_outputs)
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

    beta_risk = [np.zeros(len(risk_groups[i])) for i in range(len(risk_groups))]
    part_func = np.zeros(len(timeslots))
    weighted_avg = np.zeros(len(timeslots))

    slope = cget_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots) #@UndefinedVariable

    not_started = True

    logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))
    #we will get overflow errors when beta goes above 710, but even 200 is completely unreasonable and means that beta will diverge. in that case, QUIT
    while abs(beta) < 200 and (abs(slope) > 0 or not_started) and abs(distance) > 0.0001: #Want positive beta, Some small limit close to zero, fix to make sure we try more than one value, stop when step size is too small
        not_started = False
        prev_slope = slope
        beta += distance
        slope = cget_slope(beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots) #@UndefinedVariable
        if slope * prev_slope < 0:
            #Different signs, we have passed the zero point, change directions and half the distance
            distance /= -2
        elif abs(slope) > abs(prev_slope):
            #If the new slope is bigger than the last, we are going in the wrong direction
            distance *= -1
        logger.debug("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))

    if abs(beta) >= 200 or np.isnan(slope):
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

def derivative(test_targets, outputs, index, beta, sigma, part_func, weighted_avg, beta_force, timeslots, risk_groups, **kwargs):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    return derivative_error(beta, sigma) * derivative_betasigma(beta, sigma, part_func, weighted_avg, beta_force, index, outputs, timeslots, risk_groups)

def cox_pre_func(net, test_inputs, test_targets, block_size):
    np.seterr(all = 'raise') #I want errors!
    np.seterr(under = 'warn') #Except for underflows, just equate them to zero...

    if (block_size == 0 or block_size == len(test_targets)):
        timeslots = generate_timeslots(test_targets)
        risk_groups = get_risk_groups(timeslots)
        return {'timeslots': timeslots, 'risk_groups': risk_groups}
    else:
        return {}

def cox_block_func(test_inputs, test_targets, block_size, outputs, block_members, timeslots = None, risk_groups = None, **kwargs):
    block_outputs = outputs[block_members]
    sigma = calc_sigma(block_outputs)
    if block_size != 0 or block_size != len(test_targets):
        timeslots = generate_timeslots(test_targets[block_members])
        risk_groups = get_risk_groups(timeslots)
        retval = {'timeslots': timeslots, 'risk_groups': risk_groups}
    else:
        retval = {}
    beta, beta_risk, part_func, weighted_avg = calc_beta(block_outputs, timeslots, risk_groups)
    beta_force = get_beta_force(beta, block_outputs, risk_groups, part_func, weighted_avg)

    retval.update({'sigma':sigma, 'beta': beta, 'beta_risk': beta_risk, 'part_func': part_func, 'weighted_avg': weighted_avg, 'beta_force': beta_force})
    return retval

