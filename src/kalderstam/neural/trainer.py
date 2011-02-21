from multiprocessing import Process, Queue
from threading import Thread

from kalderstam.neural.activation_functions import logsig, tanh
from kalderstam.neural import training_functions
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.matlab_functions import plotroc, plot2d2c, stat
import matplotlib.pyplot as plt
import numpy
import logging

class Builder(Process):
    def __init__(self):
        Process.__init__(self)
        
        self.input_number = 2
        self.hidden_number = 2
        self.output_number = 1
        self.hidden_activation_function = tanh()
        self.output_activation_function = logsig()
        self.training_method = training_functions.traingd
        self.epochs = 10
        self.block_size = 1
        self.inputs = []
        self.outputs = []
    
    def run(self):
        net = build_feedforward(self.input_number, self.hidden_number, self.output_number, self.hidden_activation_function, self.output_activation_function)
        
        #if self.trainingbox.props.sensitive:
        self.net = self.training_method(net, self.inputs, self.outputs, self.epochs, self.block_size)
        
        Y = self.net.sim(self.inputs)
        
        [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, self.outputs)
        
        plotroc(Y, self.outputs)
        plot2d2c(self.net, self.inputs, self.outputs)
        plt.show()
        
class Evaluator(Process):
    def __init__(self, input, target, net_queue, error_queue):
        Process.__init__(self)
        
        self.input = input
        self.target = target
        self.net_queue = net_queue
        self.error_queue = error_queue
        
    def run(self):
        input_array = self.input
        output_array = self.target
        while True:
            net = self.net_queue.get()
            error = 0
            for i in range(0, len(input_array)):
                input = input_array[i]
                # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                output = numpy.append(numpy.array([]), output_array[i])
                #Calc output
                result = net.update(input)
                #calc sum-square error
                error += ((output - result)**2).sum()
            self.error_queue.put((net, error))
        
        
class Evaluation_Pool():
    def __init__(self, input, target, num = 8):
        self.processes = []
        self.net_queue = Queue()
        self.error_queue = Queue()
        #Build process evaluators
        for _ in range(num):
            p = Evaluator(input, target, self.net_queue, self.error_queue)
            self.processes.append(p)
            p.start()
            
    def calc_top_nets(self, net_list, num = 5):
        for net in net_list:
            self.net_queue.put(net)
        
        #Get errors
        net_errors = []
        while len(net_errors) < len(net_list):
            net_errors.append(self.error_queue.get())
        
        #compare with five best this generation
        best = []
        for net, error in net_errors:
            cmp_net = net
            cmp_error = error
            for i in range(num):
                #If nothing has been added here yet
                if len(best) < i + 1:
                    best.append((cmp_net, cmp_error))
                elif cmp_error < best[i][1]: #Shift which net we are moving through the list
                    tmp_net, tmp_error = best[i]
                    best[i] = (cmp_net, cmp_error)
                    cmp_net, cmp_error = tmp_net, tmp_error
                    
        return best
        
            
    def terminate(self):
        for p in self.processes:
            p.terminate()
    def join(self):
        for p in self.processes:
            p.join()
            
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c,\
        loadsyn2, loadsyn3
        from kalderstam.util.filehandling import parse_file, save_network
        import matplotlib.pyplot as plt
    except:
        pass
        
    P, T = loadsyn3(100)
    #P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Ecg1664_trn.dat", 39, ignorecols = 40)
        
    net_list = []        
    for _ in range(100):
        net_list.append(build_feedforward(2, 3, 1))
    pool = Evaluation_Pool(P, T)
    
    best_nets = pool.calc_top_nets(net_list, num=5)
    
    #best = train_evolutionary(net, P, T, epochs/5, random_range=5)
    #Y = best.sim(P)
    #[num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    #plot2d2c(best, P, T, 2)
    #plt.title("Only Genetic\n Total performance = " + str(total_performance) + "%")
    
    #plotroc(Y, T)
    #plt.show()
    
        
        