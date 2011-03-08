import logging
import numpy
from random import sample, random, uniform
from kalderstam.neural.matrix_network import build_feedforward, network,\
    pad_input
from kalderstam.neural.error_functions import sum_squares

logger = logging.getLogger('kalderstam.neural.training_functions')

def traingd_block(net, input_array, target_array, epochs = 300, learning_rate = 0.1, block_size = 1, momentum = 0.0, error_derivative = sum_squares.derivative):
    """Train using Gradient Descent."""
    
    def node_input(node):
        input_to_node = net.weights[node]*input
        input_to_node[node] = net.weights[node, node] #Correct for bias, which must have 1 as input
        return input_to_node.sum()
    
    for epoch in range(0, int(epochs)):
        #Iterate over training data
        logger.debug('Epoch ' + str(epoch))
        #error_sum = 0
        block_size = int(block_size)
        if block_size < 1 or block_size > len(input_array): #if 0, then equivalent to batch. 1 is equivalent to online
            block_size = len(input_array)
        
        for block in range(int(len(input_array) / block_size)):
            
            weight_corrections = numpy.zeros_like(net.weights)
            
            #Train in random order
            for i, t in sample(zip(input_array, target_array), block_size):
                # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                target = numpy.append(numpy.array([]), t)

                #prepare input
                input = pad_input(net, i)
                #Calc output
                result = net.update(input)
                
                #Now, calculate weight corrections
                #First, set gradients to zero
                error_gradients = numpy.zeros_like(net.weights[0])
                #Then, set output gradients
                error_gradients[net.output_layer] = error_derivative(target, result)
                #Now, iterate over the layers backwards, updating the gradients and weight updates in the process
                layers = list(net.layers) #Important, so we make a copy of the list. Reversing the lsit will make mayhem otherwise
                layers.reverse()
                for rows, back_rows in zip(layers[:-1], layers[1:]): #skip the input layer
                    error_gradients[back_rows] = numpy.dot(net.weights[min(rows):(max(rows) + 1), back_rows].transpose(), error_gradients[rows])
                    
                    gradient = numpy.matrix([net.activation_functions[node].derivative(node_input(node)) for node in rows] * error_gradients[rows]).T.A #transposed and back to array
                    """Bias calculation is wrong, because it doesn't remove the previous erroneous bias. just adds to it!"""
                    weight_corrections[min(rows):(max(rows) + 1), back_rows] += learning_rate * gradient * input[back_rows]
                    weight_corrections[min(rows):(max(rows) + 1), rows] += learning_rate * gradient * net.weights[min(rows):(max(rows) + 1), rows] #bias
                
            #Apply the weight updates
            net.weights += weight_corrections/block_size

    return net
            
def train_evolutionary(net, input_array, output_array, epochs = 300, population_size = 50, mutation_chance = 0.05, random_range = 3, top_number = 5, error_function = sum_squares.total_error, *args):
    """Creates more networks and evolves the best it can."""
    #Create a population of 50 networks
    best = None
    best_error = None
    population = [build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes)) for each in range(int(population_size))]
    #For each generation
    for generation in range(int(epochs)):
        error = {} #reset errors
        top_networks = [None for each in range(int(top_number))] #reset top five
        #For all networks, simulate, measure their error, and save the best network so far
        for member in population:
            error[member] = 0
            for input, output in zip(input_array, output_array):
                # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
                output = numpy.append(numpy.array([]), output)
                #Calc output
                result = member.update(input)
                #calc sum-square error
                error[member] += error_function(output, result)
            error[member] /= len(output_array)
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
                hidden = node(mother_node.function, random_range)
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
                output = node(mother_node.function)
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
                hidden = node(mother_node.function, random_range)
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
                output = node(mother_node.function)
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
            
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    
    try:
        from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
        loadsyn2, loadsyn3
        from kalderstam.util.filehandling import parse_file, save_network
        import time
        import matplotlib.pyplot as plt
    except:
        pass
        
    P, T = loadsyn3(100)
    #P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Ecg1664_trn.dat", 39, ignorecols = 40)
                
    net = build_feedforward(2, 3, 1)
    net.fix_layers()
    
    epochs = 100
    
    start = time.time()
    best = traingd_block(net, P, T, epochs, block_size = 100)
    print("traingd_block time: " + str(time.time()-start))
    Y = best.sim([pad_input(best, Pv) for Pv in P])
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plot2d2c(best, P, T, 3)
    plt.title("Gradient Descent block size 100\n Total performance = " + str(total_performance) + "%")
    
#    start = time.clock()
#    best = train_evolutionary_sequential(net, P, T, epochs / 5, random_range = 5, block_size = 0)
#    stop = time.clock()
#    Y = best.sim(P)
#    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
#    plot2d2c(best, P, T, 4)
#    plt.title("Only Genetic Sequential\n Total performance = " + str(total_performance) + "%")
#    print("Sequential time: " + str(stop-start))
#    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/test_genetic.ann")
#    
#    #net = build_feedforward(2, 4, 1)
#    
#    best = traingd_block(best, P, T, epochs, block_size = 100)
#    Y = best.sim(P)
#    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
#    plot2d2c(best, P, T, 5)
#    plt.title("Genetic Sequential followed by Gradient Descent block size 10\n Total performance = " + str(total_performance) + "%")
    
    #plotroc(Y, T)
    plt.show()
    
