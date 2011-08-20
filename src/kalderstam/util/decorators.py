import time

#These are global, and are needed for advanced benchmarking
_bench_runs = []
_bench_name = None
_bench_time = None
_bench_run = None

def benchmark(func):
    """
    A decorator that print the time of function take
    to execute.
    """
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print(func.__name__, time.time() - t)
        return res
    return wrapper

def benchmark_adv(func):
    '''
    An advanced benchmarker. Will save the time required for nested decorated functions.
    '''
    def wrapper(*args, **kwargs):
        global _bench_runs, _bench_run, _bench_name, _bench_time
        #Remember outer function
        prev_name, prev_time = _bench_name, _bench_time
        #If not None
        if _bench_run:
            _bench_run[prev_name] += time.time() - _prev_time
        else:
            _bench_run = {}
        #Add this function to run
        _bench_run[func.__name__] = 0
        #Set as current
        _bench_name, _bench_time = func.__name__, time.time()
        #Execute
        res = func(*args, **kwargs)
        #Time it
        _bench_run[_bench_name] += time.time() - _bench_time
        #Save if last, and restore
        if not prev_name:
            _bench_runs.append(_bench_run)
            _bench_run = None
        #Restore to outer function again
        _bench_name, _bench_time = prev_name, time.time()
        #Return result
        return res
    return wrapper