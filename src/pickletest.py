from multiprocessing import Process
from multiprocessing.queues import Queue

class Worker(Process):
    def __init__(self):
        Process.__init__(self)
        self.queue = Queue()

    def run(self):
        element = self.queue.get()
        print element
        print element.color
        element.method()
        

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
        
class engine():
    def __init__(self):
        self.power = "V6 800BHP"


if __name__ == '__main__':
    w = Worker()
    w.start()
    # To trigger the problem, any non-pickleable object is to be passed here.
    w.queue.put(car())
    w.join()