"""Total error = E, and derivative equals ej, where:
    E = sum of all Ej
    Ej = 1/2 * (ej)^2
    ej = result(j) - target(j)."""

def total_error(target, result):
    try:
        return ((result[:, 0] - target[:, 0]) ** 2).sum() / 2
    except FloatingPointError as e:
        print(target, result)
        raise e

def derivative(targets, results, index):
    """dE/dej = ej = result - target."""
    return (results[index, 0] - targets[index, 0])
