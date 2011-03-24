from numpy import log, exp
import numpy

shift = 4 #Also known as Delta, it's the handwaving variable.
#sigma = None #Must be calculated before training begins
#beta = None #Must be calculated before training begins

def __derivative_error__(beta, sigma):
    return -(exp(shift - beta*sigma))/(1 + exp(shift - beta*sigma))

def __derivative_betasigma__(beta, sigma, output_index, outputs, timeslots):
    """Derivative of (Beta*Sigma) with respect to y(i)"""
    return __derivative_beta__(beta, output_index, outputs, timeslots)*sigma + beta*__derivative_sigma__(sigma, output_index, outputs)

def __derivative_sigma__(sigma, output_index, outputs):
    """Eq. 12, derivative of Sigma with respect to y(i)"""
    output = outputs[output_index]
    return (output - outputs.mean())/(len(outputs)*sigma)

def __derivative_beta__(beta, output_index, outputs, timeslots):
    """Eq. 14, derivative of Beta with respect to y(i)"""
    return -__derivative_force_by_y__(beta, output_index, outputs, timeslots)/__derivative_force_by_beta__(beta, outputs, timeslots)

def __derivative_force_by_beta__(beta, outputs, timeslots):
    """Eq. 15 (and including 16), derivative of F with respect to Beta"""
    result = 0
    for s in timeslots:
        risk_outputs = get_risk_outputs(s, timeslots, outputs)
        result += -(exp(beta*risk_outputs)*risk_outputs**2).sum()/__partition_function__(beta, risk_outputs) + (__weighted_average__(beta, risk_outputs))**2
    return -result

def __derivative_force_by_y__(beta, output_index, outputs, timeslots):
    """Eq. 17, derivative of F with respect to y(i)"""
    output = outputs[output_index]
    result = 0
    for s in timeslots:
        kronicker = 0
        if s == output_index:
            kronicker = 1
        result += (kronicker - __deritative_weighted_avg__(beta, output, get_risk_outputs(s, timeslots, outputs)))
    return result

def __deritative_weighted_avg__(beta, output, risk_outputs):
    """Eq. 18. derivative of the __weighted_average__ with respect to y(i)"""
    return exp(beta*output)/__partition_function__(beta, risk_outputs)*(1 + beta*(output - __weighted_average__(beta, risk_outputs)))
    
def __weighted_average__(beta, risk_outputs):
    return (exp(beta*risk_outputs)*risk_outputs).sum()/__partition_function__(beta, risk_outputs)
    
def __partition_function__(beta, risk_outputs):
    return (exp(beta*risk_outputs)).sum() #Multiply each element in risk_outputs with Beta, then let numpy sum it up for us

def calc_beta(outputs, timeslots):
    """Find the likelihood maximizing Beta numerically."""
    beta = 1 #Start with 1
    distance = 2.0 #No idea what to start with
    def get_slope(beta, outputs, timeslots):
        result = 0
        for s in timeslots:
            output = outputs[s]
            risk_outputs = get_risk_outputs(s, timeslots, outputs)
            result += (output - __weighted_average__(beta, risk_outputs))
        return result
            
    slope = get_slope(beta, outputs, timeslots)
    
    while abs(slope) > 0.0001 : #Some small limit close to zero
        print("Beta: " + str(beta) + ", Slope: " + str(slope) + ", distance: " + str(distance))
        prev_slope = slope
        beta += distance
        slope = get_slope(beta, outputs, timeslots)
        if slope*prev_slope < 0:
            #Different signs, we have passed the zero point, change directions and half the distance
            distance /= -2
        elif (slope > 0 and slope < prev_slope) or (slope < 0 and slope > prev_slope):
            #Do nothing, we are on the right track
            pass
        else:
            #We are heading in the wrong direction, change
            distance *= -1
    
    return beta

def calc_sigma(outputs):
    """Standard deviation, just use numpy for it. need ALL results, from net.sim(inputs)"""
    return outputs.std()

def get_risk_outputs(s, timeslots, outputs):
    """s corresponds to the index of an output in outputs"""
    risk_outputs = []
    in_risk = False
    for index in timeslots:
        if s == index:
            in_risk = True #Will make sure that events that come after this are added to the risk group
        if in_risk:
            risk_outputs.append(outputs[index])
    return numpy.array(risk_outputs)

def total_error(target, result, inputs, beta, sigma):
    """E = ln(1 + exp(Delta - Beta*Sigma))."""
    return log(1 + exp(shift - beta*sigma))

def derivative(beta, sigma, output_index, outputs, timeslots):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    output = outputs[output_index]
    return __derivative_error__(beta, sigma)*__derivative_betasigma__(beta, sigma, output_index, outputs, timeslots)

if __name__ == '__main__':
    outputs = [[i*2] for i in range(4)]
    timeslots = range(len(outputs))
    print(calc_beta(numpy.array(outputs), timeslots))