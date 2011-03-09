from multiprocessing import Pool
import numpy
import os

def f(x):
    print('process id:' + str(os.getpid()) + " = " + (str(x*x)))
    return x*x

def g((x, y)):
    #print(x, y)
    print('process id:' + str(os.getpid()) + " = " + (str(x*y)))
    return x*y

def ars((a, b)):
    #print(a, b)
    result = numpy.dot(a, b)
    print('process id:' + str(os.getpid()) + " = " + (str(result)))
    return result

if __name__ == '__main__':
    p = Pool()
    result = p.map(f, range(10))
    print(result)
    
    x = range(10)
    y = range(10,20)
    result = p.map(g, zip(x, y))
    print(result)
    
    a = [numpy.arange(i) for i in range(1,10)]
    b = [numpy.arange(i, 2*i) for i in range(1,10)]
    result = p.map(ars, zip(a, b))
    print(result)