"""Total error = E, and derivative equals ej, where:
    E = sum of all Ej
    Ej = 1/2 * (ej)^2
    ej = target(j) - output(j)."""

def total_error(target, result):
    return ((target - result) ** 2).sum()/2

def derivative(target, result):
    """dE/dej = ej = target - result."""
    return (target - result)