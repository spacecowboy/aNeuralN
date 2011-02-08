import logging
import numpy
from random import sample, random, uniform
from kalderstam.neural.network import build_feedforward, node, network

logger = logging.getLogger('kalderstam.neural.network')

def traingd(net, input_array, output_array, epochs=300, learning_rate=0.1):
    """Train using Gradient Descent."""
    
    for j in range(0, int(epochs)):
        #Iterate over training data
        logger.debug('Epoch ' + str(j))
        error_sum = 0
        #Train in random order
        for i in sample(range(len(input_array)), len(input_array)):
            input = input_array[i]
            # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
            output = numpy.append(numpy.array([]), output_array[i])
                
            #Calc output
            result = net.update(input)
                
            #Set error to 0 on all nodes first
            for node in net.get_all_nodes():
                node.error_gradient = 0
            
            #Set errors on output nodes first
            for output_index in range(0, len(net.output_nodes)):
                node = net.output_nodes[output_index]
                node.error_gradient = output[output_index] - result[output_index]
                error_sum += ((output - result)**2).sum()
            
            #Iterate over the nodes and correct the weights
            for node in net.output_nodes + net.hidden_nodes:
                #Calculate local error gradient
                node.error_gradient *= node.activation_derivative(node.input_sum(input))
                #Propagate the error backwards and then update the weight
                for back_node, back_weight in node.weights.iteritems():
                    try:
                        index = int(back_node)
                        node.weights[back_node] = back_weight + learning_rate * node.error_gradient * input[index]
                        #print 'Input value used: ' + str(weight) + '*' + str(inputs[index])
                    except ValueError:
                        back_node.error_gradient += back_weight * node.error_gradient
                        node.weights[back_node] = back_weight + learning_rate * node.error_gradient * back_node.output(input)
        #normalize error
        error_sum /= len(net.output_nodes)
        logger.debug("Error = " + str(error_sum))
    return net

def traingd_batch(net, input_array, output_array, epochs=300, learning_rate=0.1):
    """Train using Gradient Descent."""
    
    for j in range(0, int(epochs)):
        #Iterate over training data
        logger.debug('Epoch ' + str(j))
        error_sum = 0
        #Train in random order
        for i in sample(range(len(input_array)), len(input_array)):
            input = input_array[i]
            # Support both [1, 2, 3] and [[1], [2], [3]] for single output node case
            output = numpy.append(numpy.array([]), output_array[i])
                
            #Calc output
            result = net.update(input)
                
            #Set error to 0 on all nodes first
            for node in net.get_all_nodes():
                node.error_gradient = 0
            
            #Set errors on output nodes first
            for output_index in range(0, len(net.output_nodes)):
                node = net.output_nodes[output_index]
                node.error_gradient = output[output_index] - result[output_index]
                error_sum += ((output - result)**2).sum()
            
            #Iterate over the nodes and correct the weights
            for node in net.output_nodes + net.hidden_nodes:
                #Calculate local error gradient
                node.error_gradient *= node.activation_derivative(node.input_sum(input))
                #Propagate the error backwards and then update the weight
                for back_node, back_weight in node.weights.iteritems():
                    try:
                        index = int(back_node)
                        node.weights[back_node] = back_weight + learning_rate * node.error_gradient * input[index]
                        #print 'Input value used: ' + str(weight) + '*' + str(inputs[index])
                    except ValueError:
                        back_node.error_gradient += back_weight * node.error_gradient
                        node.weights[back_node] = back_weight + learning_rate * node.error_gradient * back_node.output(input)
        #normalize error
        error_sum /= len(net.output_nodes)
        logger.debug("Error = " + str(error_sum))
    return net
            
def train_evolutionary(net, input_array, output_array, epochs=300, population_size = 50, mutation_chance = 0.05, random_range=3):
    """Creates more networks and evolves the best it can."""
    #Create a population of 50 networks
    population = list()
    best = None
    best_error = None
    for index in range(population_size):
        population.append(build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes), net.hidden_function, net.output_function))
    #For each generation
    for generation in range(int(epochs)):
        error = dict() #reset errors
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
            
            population[child_index] = network(mother.hidden_function, mother.output_function)
            population[child_index].num_of_inputs = mother.num_of_inputs
    
            #Input nodes
            #for i in range(len(mother.input_nodes)):
            #    input = input_node()
            #    population[child_index].input_nodes.append(input)
            #input_nodes = population[child_index].input_nodes
            
            #Hidden layer
            for i in range(len(mother.hidden_nodes)):
                hidden = node(mother.hidden_function, random_range)
                weights = dict()
                for j in range(mother.num_of_inputs):
                    choice = sample([mother, father], 1)[0]
                    weights[j] = choice.hidden_nodes[i].weights[j]
                    if (random() < mutation_chance): # mutation chance
                        weights[j] += uniform(-random_range, random_range)
                
                hidden.connect_nodes(range(mother.num_of_inputs), weights)
                population[child_index].hidden_nodes.append(hidden)
            hidden_nodes = population[child_index].hidden_nodes
                
            #Output nodes
            for i in range(len(mother.output_nodes)):
                output = node(mother.output_function)
                weights = dict()
                for j in range(len(mother.hidden_nodes)):
                    choice = sample([mother, father], 1)[0]
                    weights[hidden_nodes[j]] = choice.output_nodes[i].weights[choice.hidden_nodes[j]]
                    if (random() < mutation_chance): # mutation chance
                        weights[hidden_nodes[j]] += uniform(-random_range, random_range)
                
                output.connect_nodes(hidden_nodes, weights)
                population[child_index].output_nodes.append(output)
        
        logger.debug("Generation " + str(generation) + ", best so far: " + str(best_error))
                
    #finally, return the best network
    return best
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c,\
        loadsyn2, loadsyn3
        from kalderstam.util.filehandling import parse_file
        import matplotlib.pyplot as plt
    except:
        pass
        
    P, T = loadsyn3(100)
    #P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Ecg1664_trn.dat", 39, 1)
                
    net = build_feedforward(2, 2, 1)
    
    epochs = 100
    
    best = traingd(net, P, T, epochs)
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plot2d2c(best, P, T, 1)
    plt.title("Only Gradient Descent.\n Total performance = " + str(total_performance) + "%")
    
    best = train_evolutionary(net, P, T, epochs/5, random_range=5)
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plot2d2c(best, P, T, 2)
    plt.title("Only Genetic\n Total performance = " + str(total_performance) + "%")
    
    best = traingd(best, P, T, epochs)
    Y = best.sim(P)
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plot2d2c(best, P, T, 3)
    plt.title("Genetic followed by Gradient Descent\n Total performance = " + str(total_performance) + "%")
    
    #net.traingd(P, T, 300, 0.1)
    
    #Y = net.sim(P)
    #Y2 = best.sim(P)
    
    #[num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    #[num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y2, T)
    
    #plotroc(Y, T)
    plt.show()
    