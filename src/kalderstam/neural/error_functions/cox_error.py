shift = 4 #Also known as Delta, it's the handwaving variable.

def total_error(target, result):
    """E = ln(1 + exp(Delta - Beta*Sigma)).
    Sigma = result.std() # Numpy standard deviation
    """
    pass

def derivative(target, result):
    """dE/d(Beta*Sigma) * d(Beta*Sigma)/dresult."""
    pass

