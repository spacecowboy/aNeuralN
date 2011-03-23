from numpy import log, exp
import numpy

shift = 4 #Also known as Delta, it's the handwaving variable.
#sigma = None #Must be calculated before training begins
#beta = None #Must be calculated before training begins
#beta_sigma = None #Beta * Sigma, unecessary to recalculate each time

def calc_betas(inputs):
    """Find the likelihood maximizing Beta numerically."""
    betas = []
    for input in inputs:
        beta = numpy.ones(len(input)) #Start with 1
        distance = 2 #No idea what to start with
        slope = None
        
        while abs(slope) > 0.00001: #Some small limit close to zero
            prev_slope = slope
            beta += distance
            slope = None #Calc new slope
            if slope*prev_slope < 0:
                #Different signs, we have passed the zero point, change directions and half the distance
                distance /= -2
            elif (slope > 0 and slope < prev_slope) or (slope < 0 and slope > prev_slope):
                #Do nothing, we are on the right track
                pass
            else:
                #We are heading in the wrong direction, change
                distance *= -1
        betas.append(beta)
    
    return numpy.array(betas) #convert to numpy array to keep things compatible

def calc_sigma(results):
    """Standard deviation, just use numpy for it. need ALL results, from net.sim(inputs)"""
    return results.std()

def __derivative_error__(inputs, result, beta, sigma):
    return -(exp(shift - beta*sigma))/(1 + exp(shift - beta*sigma))

def __derivative_betasigma__(result, results, beta, sigma):
    """dBeta/dY*Sigma + Beta*dSigma/dYi"""
    n = len(results)
    """dSigma/dYi"""
    sigma_deriv = (result - results.mean)/(n*sigma)
    
    
    beta_deriv = None
    pass
    return beta_deriv*sigma + beta*sigma_deriv


def total_error(target, result, inputs, beta, sigma):
    """E = ln(1 + exp(Delta - Beta*Sigma))."""
    return log(1 + exp(shift - beta*sigma))

def derivative(target, result, inputs, beta, sigma):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    return __derivative_error__(inputs, result, beta, sigma)*__derivative_betasigma__(beta, sigma)

if __name__ == '__main__':
    inputs = [[i] for i in range(4)]
    print(calc_betas(numpy.array(inputs)))