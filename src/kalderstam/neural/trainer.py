from multiprocessing import Process

from kalderstam.neural.functions.activation_functions import logsig, tanh
from kalderstam.neural.functions import training_functions
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.matlab_functions import plotroc, plot2d2c, stat
import matplotlib.pyplot as plt

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
        self.inputs = []
        self.outputs = []
    
    def run(self):
        net = build_feedforward(self.input_number, self.hidden_number, self.output_number, self.hidden_activation_function, self.output_activation_function)
        
        #if self.trainingbox.props.sensitive:
        self.net = self.training_method(net, self.inputs, self.outputs, self.epochs)
        
        Y = self.net.sim(self.inputs)
        
        [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, self.outputs)
        
        plotroc(Y, self.outputs)
        plot2d2c(self.net, self.inputs, self.outputs)
        plt.show()