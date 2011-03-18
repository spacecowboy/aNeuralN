import logging
import numpy
from random import sample, random, uniform
from kalderstam.neural.network import build_feedforward, node, network
from kalderstam.neural.error_functions import sum_squares
from kalderstam.util.filehandling import get_validation_set

logger = logging.getLogger('kalderstam.neural.training_functions')

def train_committee(com, train_func, input_array, target_array, *train_args, **train_kwargs):
    return mp_train_committee(com, train_func, input_array, target_array, *train_args, **train_kwargs)

def traingd(net, input_array, output_array, epochs = 300, learning_rate = 0.1, error_derivative = sum_squares.derivative):
    """Train using Gradient Descent."""
    
    for epoch in range(0, int(epochs)):
        #Iterate over training data
        logger.debug('Epoch ' + str(epoch))
        error_sum = 0
        #Train in random order
        for input, output in sample(zip(input_array, output_array), len(input_array)):
            # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
            output = numpy.append(numpy.array([]), output)
                
            #Calc output
            result = net.update(input)
                
            #Set error to 0 on all nodes first
            for node in net.get_all_nodes():
                node.error_gradient = 0
            
            #Set errors on output nodes first
            for node, gradient in zip(net.output_nodes, error_derivative(output, result)):
                node.error_gradient = gradient
                error_sum += (gradient ** 2).sum()
            
            #Iterate over the nodes and correct the weights
            for node in net.output_nodes + net.hidden_nodes:
                #Calculate local error gradient
                node.error_gradient *= node.activation_derivative(node.input_sum(input))
                #Propagate the error backwards and then update the weight
                for back_node, back_weight in node.weights.items():
                    try:
                        index = int(back_node)
                        node.weights[back_node] = back_weight + learning_rate * node.error_gradient * input[index]
                        #print 'Input value used: ' + str(weight) + '*' + str(input[index])
                    except ValueError:
                        back_node.error_gradient += back_weight * node.error_gradient
                        node.weights[back_node] = back_weight + learning_rate * node.error_gradient * back_node.output(input)
                #Finally, bias
                node.bias = node.bias + learning_rate * node.error_gradient * node.bias
                
        #normalize error
        error_sum /= len(net.output_nodes)
        error_sum /= len(output_array)
        logger.debug("Error = " + str(error_sum))
    return net

def traingd_block(net, (test_inputs, test_targets), (validation_inputs, validation_targets), epochs = 300, learning_rate = 0.1, block_size = 1, momentum = 0.0, error_derivative = sum_squares.derivative, error_function = sum_squares.total_error, stop_error_value = 0):
    """Train using Gradient Descent."""
    
#    targets = []
#    for output in test_targets:
#        # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
#        targets.append(output)
#    targets = numpy.array(targets)
    
    for epoch in range(0, int(epochs)):
        #Iterate over training data
        logger.debug('Epoch ' + str(epoch))
        #error_sum = 0
        block_size = int(block_size)
        if block_size < 1 or block_size > len(test_inputs): #if 0, then equivalent to batch. 1 is equivalent to online
            block_size = len(test_inputs)
        
        for block in range(int(len(test_inputs) / block_size)):
            
            #Set error to 0 on all nodes first
            for node in net.get_all_nodes():
                node.weight_corrections = {}
            
            #Train in random order
            for input, target in sample(zip(test_inputs, test_targets), block_size):
                # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
#                target = numpy.append(numpy.array([]), output)

                #Calc output
                result = net.update(input)

                #Set error to 0 on all nodes first
                for node in net.get_all_nodes():
                    node.error_gradient = 0

                #Set errors on output nodes first
                for node, gradient in zip(net.output_nodes, error_derivative(target, result)):
                    node.error_gradient = gradient
                
                #Iterate over the nodes and correct the weights
                for node in net.output_nodes + net.hidden_nodes:
                    #Calculate local error gradient
                    node.error_gradient *= node.activation_function.derivative(node.input_sum(input))

                    #Propagate the error backwards and then update the weights
                    for back_node, back_weight in node.weights.items():
                        
                        if back_node not in node.weight_corrections:
                            node.weight_corrections[back_node] = []
                            
                        try:
                            index = int(back_node)
                            node.weight_corrections[back_node].append(node.error_gradient * input[index])
                        except ValueError:
                            back_node.error_gradient += back_weight * node.error_gradient
                            node.weight_corrections[back_node].append(node.error_gradient * back_node.output(input))
                    
                    #Finally, correct the bias
                    if "bias" not in node.weight_corrections:
                        node.weight_corrections["bias"] = []
                    node.weight_corrections["bias"].append(node.error_gradient * node.bias)
            
            #Iterate over the nodes and correct the weights
            for node in net.output_nodes + net.hidden_nodes:
                #Calculate weight update
                for back_node, back_weight in node.weights.items():
                    node.weights[back_node] = back_weight + learning_rate * sum(node.weight_corrections[back_node]) / len(node.weight_corrections[back_node])
                #Don't forget bias
                node.bias = node.bias + learning_rate * sum(node.weight_corrections["bias"]) / len(node.weight_corrections["bias"])
                
        #Calculate error of the network and print
        
        if len(test_inputs > 0):
            test_results = net.sim(test_inputs)
            test_error = error_function(test_targets, test_results)/len(test_targets)
            logger.debug("Test Error = " + str(test_error))
            if test_error <= stop_error_value:
                break
            
        if validation_inputs != None and len(validation_inputs) > 0:
            validation_results = net.sim(validation_inputs)
            validation_error = error_function(validation_targets, validation_results)/len(validation_targets)
            logger.debug("Validation Error = " + str(validation_error))
            if validation_error <= stop_error_value:
                break

    return net
            
def train_evolutionary(net, (input_array, output_array), (validation_inputs, validation_targets), epochs = 300, population_size = 50, mutation_chance = 0.05, random_range = 3, top_number = 5, error_function = sum_squares.total_error, *args):
    """Creates more networks and evolves the best it can.
    Does NOT use any validation set..."""
    #Create a population of 50 networks
    best = None
    best_error = None
    population = [build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes)) for each in range(int(population_size))]
    #For each generation
    for generation in range(int(epochs)):
        error = {} #reset errors
        top_networks = [None for each in range(int(top_number))] #reset top five
        #For all networks, simulate, measure their error, and save the best network so far
        #sim_results = mp_nets_sim(population, input_array)
        sim_results = [net.sim(input_array) for net in population]
        for member, sim_result in zip(population, sim_results):
#            error[member] = 0
#            for input, output in zip(input_array, output_array):
#                # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
#                output = numpy.append(numpy.array([]), output)
#                #Calc output
#                result = member.update(input)
#                #calc sum-square error
#                error[member] += error_function(output, result)
            #error[member] = error_function(output_array, member.sim(input_array))/len(output_array)
            error[member] = error_function(output_array, sim_result)/len(output_array)
            #compare with best
            if not best or error[member] < best_error:
                best = member
                best_error = error[member]
            #compare with five best this generation
            comp_net = member
            for i in range(0, len(top_networks)):
                if top_networks[i] and error[comp_net] < error[top_networks[i]]:
                    old = top_networks[i]
                    top_networks[i] = comp_net
                    comp_net = old
                elif not top_networks[i]:
                    top_networks[i] = comp_net
                    break
                
        #Select the best 5 networks, mate them randomly and create 50 new networks
        for child_index in range(len(population)):
            [mother, father] = sample(top_networks, 2)
            
            population[child_index] = network()
            population[child_index].num_of_inputs = mother.num_of_inputs
            
            #Hidden layer
            for mother_node, father_node in zip(mother.hidden_nodes, father.hidden_nodes):
                hidden = node(mother_node.activation_function, random_range)
                weights = {}
                for input_number in range(mother.num_of_inputs):
                    choice = sample([mother_node, father_node], 1)[0]
                    weights[input_number] = choice.weights[input_number]
                    if (random() < mutation_chance): # mutation chance
                        weights[input_number] += uniform(-random_range, random_range)
                #Don't forget bias
                if (random() < mutation_chance): # mutation chance
                    hidden.bias = uniform(-random_range, random_range)
                else:
                    hidden.bias = sample([mother_node, father_node], 1)[0].bias
                
                hidden.connect_nodes(range(mother.num_of_inputs), weights)
                population[child_index].hidden_nodes.append(hidden)
                
            hidden_nodes = population[child_index].hidden_nodes
                
            #Output nodes
            #for mother_node, father_node in zip(mother.output_nodes, father.output_nodes):
            for output_number in range(len(mother.output_nodes)):
                output = node(mother_node.activation_function)
                weights = {}
                for hidden_number in range(len(mother.hidden_nodes)):
                    choice = sample([mother, father], 1)[0]
                    weights[hidden_nodes[hidden_number]] = choice.output_nodes[output_number].weights[choice.hidden_nodes[hidden_number]]
                    if (random() < mutation_chance): # mutation chance
                        weights[hidden_nodes[hidden_number]] += uniform(-random_range, random_range)
                #Don't forget bias
                if (random() < mutation_chance): # mutation chance
                    output.bias = uniform(-random_range, random_range)
                else:
                    output.bias = sample([mother, father], 1)[0].output_nodes[output_number].bias
                
                output.connect_nodes(hidden_nodes, weights)
                population[child_index].output_nodes.append(output)
        
        logger.debug("Generation " + str(generation) + ", best so far: " + str(best_error))
                
    #finally, return the best network
    return best

def train_evolutionary_sequential(net, input_array, output_array, epochs = 300, population_size = 50, mutation_chance = 0.05, random_range = 3, top_number = 5, error_function = sum_squares.total_error, block_size = 0, *args):
    """Creates more networks and evolves the best it can."""
    #Create a population of 50 networks
    best_error = None
    if block_size == 0 or block_size > population_size:
        block_size = population_size
    population = [build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes)) for each in range(int(population_size))]
    #For each generation
    for generation in range(int(epochs)):
        for mating in range(len(population)):
            error = {}
            ranking = []
            #2 Networks mate and kill a third
            selection = sample(population, 3)
            for net in selection:
                error[net] = 0
                #Calculate fitness
                for input, output in sample(zip(input_array, output_array), block_size):
                    # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                    output = numpy.append(numpy.array([]), output)
                    #Calc output
                    result = net.update(input)
                    #calc sum-square error
                    error[net] += error_function(output, result)
                error[net] /= block_size
                if best_error == None or error[net] < best_error:
                    best_error = error[net]
                    
                if len(ranking) < 1:
                    ranking.append(net)
                else:
                    cmp_net = net
                    for index in range(len(ranking)):
                        if error[cmp_net] < error[ranking[index]]:
                            tmp_net = ranking[index]
                            ranking[index] = cmp_net
                            cmp_net = tmp_net
                    ranking.append(cmp_net)
            #Now mate the two first networks, and destroy the third
            mother = ranking[0]
            father = ranking[1]
            child_index = population.index(ranking[2])
            
            population[child_index] = network()
            population[child_index].num_of_inputs = mother.num_of_inputs
            
            #Hidden layer
            for mother_node, father_node in zip(mother.hidden_nodes, father.hidden_nodes):
                hidden = node(mother_node.activation_function, random_range)
                weights = {}
                for input_number in range(mother.num_of_inputs):
                    choice = sample([mother_node, father_node], 1)[0]
                    weights[input_number] = choice.weights[input_number]
                    if (random() < mutation_chance): # mutation chance
                        weights[input_number] += uniform(-random_range, random_range)
                #Don't forget bias
                if (random() < mutation_chance): # mutation chance
                    hidden.bias = uniform(-random_range, random_range)
                else:
                    hidden.bias = sample([mother_node, father_node], 1)[0].bias
                
                hidden.connect_nodes(range(mother.num_of_inputs), weights)
                population[child_index].hidden_nodes.append(hidden)
                
            hidden_nodes = population[child_index].hidden_nodes
                
            #Output nodes
            #for mother_node, father_node in zip(mother.output_nodes, father.output_nodes):
            for output_number in range(len(mother.output_nodes)):
                output = node(mother_node.activation_function)
                weights = {}
                for hidden_number in range(len(mother.hidden_nodes)):
                    choice = sample([mother, father], 1)[0]
                    weights[hidden_nodes[hidden_number]] = choice.output_nodes[output_number].weights[choice.hidden_nodes[hidden_number]]
                    if (random() < mutation_chance): # mutation chance
                        weights[hidden_nodes[hidden_number]] += uniform(-random_range, random_range)
                #Don't forget bias
                if (random() < mutation_chance): # mutation chance
                    output.bias = uniform(-random_range, random_range)
                else:
                    output.bias = sample([mother, father], 1)[0].output_nodes[output_number].bias
                
                output.connect_nodes(hidden_nodes, weights)
                population[child_index].output_nodes.append(output)
        
        logger.debug("Generation " + str(generation) + ", best so far (block_error): " + str(best_error))
                
    #finally, return the best network
    best = None
    best_error = None
    for net in population:
        net_error = 0
        #Calculate fitness
        for input, output in zip(input_array, output_array):
            # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
            output = numpy.append(numpy.array([]), output)
            #Calc output
            result = net.update(input)
            #calc sum-square error
            net_error += error_function(output, result)
        net_error /= len(output_array)
        
        if best_error == None or net_error < best_error:
            best_error = net_error
            best = net
    logger.debug("Generation " + str(generation + 1) + ", best found (total error): " + str(best_error))
    return best

from kalderstam.neural.mp_network import mp_net_sim_inputs, mp_nets_sim,\
    mp_train_committee
            
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    
    try:
        from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
        loadsyn2, loadsyn3, plotroc
        from kalderstam.util.filehandling import parse_file, save_network
        from kalderstam.util.decorators import benchmark
        from kalderstam.neural.network import build_feedforward_committee
        import time
        import matplotlib.pyplot as plt
    except:
        pass
        
    P, T = loadsyn3(100)
    p, t = P, T
    #P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Ecg1664_trn.dat", 39, ignorecols = 40)
    test, validation = get_validation_set(P, T)
    net = build_feedforward(2, 6, 1)
    
    epochs = 100
    
#    best = traingd(net, P, T, epochs)
#    Y = best.sim(P)
#    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
#    plot2d2c(best, P, T, 1)
#    plt.title("Only Gradient Descent.\n Total performance = " + str(total_performance) + "%")
    
    #start = time.clock()
    best = benchmark(train_evolutionary)(net, test, validation, epochs/10, random_range = 5)
    #stop = time.clock()
    P, T = test
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plot2d2c(best, P, T, 1)
    plt.title("Only Genetic\n [Test set] Total performance = " + str(total_performance) + "%")
    #print("Genetic time: " + str(stop-start))
    P, T = validation
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plt.title("Only Genetic\n [Validation set] Total performance = " + str(total_performance) + "%")
    #print("Genetic time: " + str(stop-start))
    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/test_genetic.ann")
    
    #net = build_feedforward(2, 4, 1)
    
    #start = time.time()
    best = benchmark(traingd_block)(best, test, validation, epochs, block_size = 10, stop_error_value = 0.1)
    #print("traingd_block time: " + str(time.time()-start))
    Y = best.sim(P)
    
    P, T = validation
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plot2d2c(best, P, T, 3)
    plt.title("Genetic followed by Gradient Descent block size 10\n [Validation set] Total performance = " + str(total_performance) + "%")
    P, T = test
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plot2d2c(best, P, T, 4)
    #plotroc(Y, T)
    plt.title("Genetic followed by Gradient Descent block size 10\n [Test set] Total performance = " + str(total_performance) + "%")

    com = build_feedforward_committee(size = 4, input_number = 2, hidden_number = 6, output_number = 1)
    
    benchmark(train_committee)(com, train_evolutionary, p, t, epochs/10, random_range = 5)
    
    Y = com.sim(p)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, t)
    plot2d2c(com, p, t, 5)
    plt.title("Committee Genetic\n [Cross validation] Total performance = " + str(total_performance) + "%")
    
    
    benchmark(train_committee)(com, traingd_block, p, t, epochs, block_size = 10, stop_error_value = 0.08)
    
    Y = com.sim(p)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, t)
    plot2d2c(com, p, t, 6)
    plt.title("Committee Genetic followed by Gradient Descent block size 10\n [Cross validation] Total performance = " + str(total_performance) + "%")
    plotroc(Y, t, 7)
    
    #plotroc(Y, T)
    plt.show()
    
