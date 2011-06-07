def benchmark(func):
    """
    A decorator that print the time of function take
    to execute.
    """
    import time
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print(func.__name__, time.time() - t)
        return res
    return wrapper
