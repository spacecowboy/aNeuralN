from numpy import log, exp

shift = 4 #Also known as Delta, it's the handwaving variable.
#sigma = None #Must be calculated before training begins
#beta = None #Must be calculated before training begins
#beta_sigma = None #Beta * Sigma, unecessary to recalculate each time

def calc_betas(inputs):
    """Find the likelihood maximising Beta numerically."""
    pass

def calc_sigma(results):
    """Standard deviation, just use numpy for it. need ALL results, from net.sim(inputs)"""
    return results.std()

def __derivative_error__(inputs, result, beta, sigma):
    return -(exp(shift - beta*sigma))/(1 + exp(shift - beta*sigma))

def __derivative_betasigma__():
    """dBeta/dY*Sigma + Beta*dSigma/dY"""
    pass

def total_error(target, result, inputs, beta, sigma):
    """E = ln(1 + exp(Delta - Beta*Sigma))."""
    return log(1 + exp(shift - beta*sigma))

def derivative(target, result, inputs, beta, sigma):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    return __derivative_error__(inputs, result, beta, sigma)*__derivative_betasigma__(beta, sigma)

