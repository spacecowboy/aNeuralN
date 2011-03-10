from multiprocessing import Process
from multiprocessing.queues import Queue
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.matlab_functions import loadsyn1

class Worker(Process):
    def __init__(self, q):
        Process.__init__(self)
        self.queue = q

    def run(self):
        element, args, kwargs = self.queue.get()
        print element
        print args
        print kwargs
        print element.color
        element.method()
        element.args_method(*args, **kwargs)
        
class Neural_Worker(Process):
    def __init__(self, q):
        Process.__init__(self)
        self.queue = q

    def run(self):
        net, args, kwargs = self.queue.get()
        Y = net.sim(*args, **kwargs)
        print Y
        
def external_method(what):
    print(what)
    
class class_method():
    def external_class_method(self, what):
        print(what)
    
class car():
    def __init__(self, f = external_method, ext_c = class_method()):
        self.color = "self.attribute works"
        self.motor = engine()
        self.ext_func = f
        self.ext_c = ext_c
        #self.ext_c_func = ext_c.external_class_method
    
    def method(self):
        print("Printing my attribute " + self.color)
        
        def inner_method(t):
            return "Inner method works! " + str(t)
        print(inner_method("Great"))
        
        print("Inner object: " + self.motor.power)
        
        self.ext_func("Testing method passing..")
        self.ext_c.external_class_method("Testing class with method passing..")
        #self.ext_c_func("Testing class method passing..")
        
        
    def args_method(self, *args, **kwargs):
        print("Args: " + str(args))
        print("Kwargs: " + str(kwargs))
        
class engine():
    def __init__(self):
        self.power = "V6 800BHP"
        



if __name__ == '__main__':
    queue = Queue()
    w = Worker(queue)
    w.start()
    # To trigger the problem, any non-pickleable object is to be passed here.
    args = ['hej', 'hopp']
    kwargs = {}
    kwargs['one'] = 1
    queue.put((car(), args, kwargs))
    w.join()
    
    #Now try with a neural network
    
    P1, T1 = loadsyn1(100)
    P2, T2 = loadsyn1(100)
                
    net = build_feedforward(2, 1, 1)
    
    nw = Neural_Worker(queue)
    nw.start()
    
    #queue.put((net, [P1], {}))
    
    nw.join()