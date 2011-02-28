
def total_error(target, result):
    return sum((target - result) ** 2)/2

def derivative(target, result):
    return (target - result)