from multiprocessing import Process, Queue
from threading import Thread

from kalderstam.neural.activation_functions import logsig, tanh
from kalderstam.neural import training_functions
from kalderstam.neural.network import build_feedforward, node, network
from kalderstam.neural.matlab_functions import plotroc, plot2d2c, stat
import matplotlib.pyplot as plt
import numpy
import logging
from random import sample, random, uniform

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
        
class Trainer(Process):
    def __init__(self):
        Process.__init__(self)
        self.stop = False
        self.logger = logging.getLogger('kalderstam.neural.training_functions')
        self.training_function = self.traingd_block #default
        self.kwargs = {}
        
    def train(self, **kwargs):
        self.kwargs = kwargs
        print(kwargs)
        self.run()
    
    def traingd_block(self, net, input_array, output_array, epochs=300, learning_rate=0.1, block_size=1, momentum=0.0):
        """Train using Gradient Descent."""
        
        for j in range(0, int(epochs)):
            if self.stop:
                print("STOP!")
                break
            #Iterate over training data
            self.logger.debug('Epoch ' + str(j))
            #error_sum = 0
            block_size = int(block_size)
            if block_size < 1 or block_size > len(input_array): #if 0, then equivalent to batch. 1 is equivalent to online
                block_size = len(input_array)
            
            for _ in range(int(len(input_array)/block_size)):
                
                #Set error to 0 on all nodes first
                for node in net.get_all_nodes():
                    node.weight_corrections = {}
                
                #Train in random order
                for i in sample(range(len(input_array)), block_size):
                    input = input_array[i]
                    output = output_array[i]
    
                    # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                    answer = numpy.append(numpy.array([]), output)
    
                    #Calc output
                    result = net.update(input)
    
                    #Set error to 0 on all nodes first
                    for node in net.get_all_nodes():
                        node.error_gradient = 0
    
                    #Set errors on output nodes first
                    for output_index in range(0, len(net.output_nodes)):
                        node = net.output_nodes[output_index]
                        node.error_gradient = answer[output_index] - result[output_index]
    
                    
                    #Iterate over the nodes and correct the weights
                    for node in net.output_nodes + net.hidden_nodes:
                        #Calculate local error gradient
                        node.error_gradient *= node.activation_derivative(node.input_sum(input))
    
                        #Propagate the error backwards and then update the weights
                        for back_node, back_weight in node.weights.iteritems():
                            
                            if back_node not in node.weight_corrections:
                                (node.weight_corrections)[back_node] = []
                                
                            try:
                                index = int(back_node)
                                node.weight_corrections[back_node].append(node.error_gradient*input[index])
                            except ValueError:
                                back_node.error_gradient += back_weight * node.error_gradient
                                node.weight_corrections[back_node].append(node.error_gradient * back_node.output(input))
                        
                        #Finally, correct the bias
                        if "bias" not in node.weight_corrections:
                            (node.weight_corrections)["bias"] = []
                            node.weight_corrections["bias"].append(node.error_gradient * node.bias)
                
                #Iterate over the nodes and correct the weights
                for node in net.output_nodes + net.hidden_nodes:
                    #Calculate weight update
                    for back_node, back_weight in node.weights.iteritems():
                        node.weights[back_node] = back_weight + learning_rate * sum(node.weight_corrections[back_node])/len(node.weight_corrections[back_node])
                    #Don't forget bias
                    node.bias = node.bias + learning_rate * sum(node.weight_corrections["bias"])/len(node.weight_corrections["bias"])
                        
    
            #normalize error
            #error_sum = sum()/len(net.output_nodes)
            #self.logger.debug("Error = " + str(error_sum))
        return net
                
    def train_evolutionary(self, net, input_array, output_array, epochs=300, population_size = 50, mutation_chance = 0.05, random_range=3.0, *args):
        """Creates more networks and evolves the best it can."""
        #Create a population of 50 networks
        #population_size = int(population_size)
        population = list()
        best = None
        best_error = None
        for _ in range(int(population_size)):
            population.append(build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes)))
        #For each generation
        for generation in range(int(epochs)):
            if self.stop:
                break
            error = {} #reset errors
            best_five = [None, None, None, None, None] #reset top five
            #For all networks, simulate, measure their error, and save the best network so far
            for member in population:
                error[member] = 0
                for i in range(0, len(input_array)):
                    input = input_array[i]
                    # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                    output = numpy.append(numpy.array([]), output_array[i])
                    #Calc output
                    result = member.update(input)
                    #calc sum-square error
                    error[member] += ((output - result)**2).sum()
                    
                #compare with best
                if not best or error[member] < best_error:
                    best = member
                    best_error = error[member]
                #compare with five best this generation
                comp_net = member
                for i in range(0, len(best_five)):
                    if best_five[i] and error[comp_net] < error[best_five[i]]:
                        old = best_five[i]
                        best_five[i] = comp_net
                        comp_net = old
                    elif not best_five[i]:
                        best_five[i] = comp_net
                        break
                    
            #Select the best 5 networks, mate them randomly and create 50 new networks
            for child_index in range(len(population)):
                [mother, father] = sample(best_five, 2)
                
                population[child_index] = network()
                population[child_index].num_of_inputs = mother.num_of_inputs
        
                #Input nodes
                #for i in range(len(mother.input_nodes)):
                #    input = input_node()
                #    population[child_index].input_nodes.append(input)
                #input_nodes = population[child_index].input_nodes
                
                #Hidden layer
                for i in range(len(mother.hidden_nodes)):
                    hidden = node(mother.hidden_nodes[i].function, random_range)
                    weights = {}
                    for j in range(mother.num_of_inputs):
                        choice = sample([mother, father], 1)[0]
                        weights[j] = choice.hidden_nodes[i].weights[j]
                        if (random() < mutation_chance): # mutation chance
                            weights[j] += uniform(-random_range, random_range)
                    #Don't forget bias
                    if (random() < mutation_chance): # mutation chance
                        hidden.bias = uniform(-random_range, random_range)
                    else:
                        hidden.bias = sample([mother, father], 1)[0].hidden_nodes[i].bias
                    
                    hidden.connect_nodes(range(mother.num_of_inputs), weights)
                    population[child_index].hidden_nodes.append(hidden)
                hidden_nodes = population[child_index].hidden_nodes
                    
                #Output nodes
                for i in range(len(mother.output_nodes)):
                    output = node(mother.output_nodes[i].function)
                    weights = {}
                    for j in range(len(mother.hidden_nodes)):
                        choice = sample([mother, father], 1)[0]
                        weights[hidden_nodes[j]] = choice.output_nodes[i].weights[choice.hidden_nodes[j]]
                        if (random() < mutation_chance): # mutation chance
                            weights[hidden_nodes[j]] += uniform(-random_range, random_range)
                    #Don't forget bias
                    if (random() < mutation_chance): # mutation chance
                        output.bias = uniform(-random_range, random_range)
                    else:
                        output.bias = sample([mother, father], 1)[0].output_nodes[i].bias
                    
                    output.connect_nodes(hidden_nodes, weights)
                    population[child_index].output_nodes.append(output)
            
            self.logger.debug("Generation " + str(generation) + ", best so far: " + str(best_error))
                    
        #finally, return the best network
        return best
    
    def run(self):
        self.net = self.training_function(**self.kwargs)
        
        Y = self.net.sim(self.kwargs['input_array'])
        
        plotroc(Y, self.kwargs['output_array'])
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
    
        
        