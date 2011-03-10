from multiprocessing import Process
from multiprocessing.queues import Queue

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
        

class car():
    def __init__(self):
        self.color = "self.attribute works"
        self.motor = engine()
    
    def method(self):
        print("Printing my attribute " + self.color)
        
        def inner_method(t):
            return "Inner method works! " + str(t)
        print(inner_method("Great"))
        
        print("Inner object: " + self.motor.power)
        
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