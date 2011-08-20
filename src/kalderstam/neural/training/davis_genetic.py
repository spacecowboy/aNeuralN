import logging
import numpy
from random import sample, random, uniform
from kalderstam.neural.network import build_feedforward, node, network, \
    connect_nodes
from kalderstam.neural.error_functions import sum_squares
import kalderstam.util.graphlogger as glogger

from kalderstam.util.decorators import benchmark_adv

logger = logging.getLogger('kalderstam.neural.training_functions')

numpy.seterr(all = 'raise') #I want errors!

@benchmark_adv
def crossover_node(mother, father):
    child = network()
    child.num_of_inputs = mother.num_of_inputs

    tr = {}
    for i in xrange(mother.num_of_inputs):
        tr[i] = i

    for mother_node, father_node in zip(mother.hidden_nodes, father.hidden_nodes):
        child_node = node(mother_node.activation_function, 1)
        tr[mother_node] = child_node
        tr[father_node] = child_node

        #choose one node to pass on
        choice = sample([mother_node, father_node], 1)[0]

        #map the weights and bias of that node to the child node
        for keynode, weight in choice.weights.items():
            child_node.weights[tr[keynode]] = weight
        child_node.bias = choice.bias

        child.hidden_nodes.append(child_node)

    for mother_node, father_node in zip(mother.output_nodes, father.output_nodes):
        child_node = node(mother_node.activation_function, 1)
        tr[mother_node] = child_node
        tr[father_node] = child_node

        #choose one node to pass on
        choice = sample([mother_node, father_node], 1)[0]

        #map the weights and bias of that node to the child node
        for keynode, weight in choice.weights.items():
            child_node.weights[tr[keynode]] = weight
        child_node.bias = choice.bias

        child.output_nodes.append(child_node)

    return child

@benchmark_adv
def mutate_biased(parent, mutation_chance, random_range):
    child = crossover_node(parent, parent)

    for node in child.get_all_nodes():
        for keynode, weight in node.weights.items():
            if (random() < mutation_chance): # mutation chance
                node.weights[keynode] += uniform(-random_range, random_range)
        if (random() < mutation_chance): # mutation chance
            node.bias += uniform(-random_range, random_range)

    return child

@benchmark_adv
def train_evolutionary(net, (input_array, output_array), (validation_inputs, validation_targets), epochs = 300, population_size = 50, mutation_chance = 0.05, random_range = 1, error_function = sum_squares.total_error, *args): #@UnusedVariable
    """Creates more networks and evolves the best it can.
    Uses validation set only for plotting.
    This version does not replace the entire population each generation. Two parents are selected at random to create a child. The two best are
    returned to the population for breeding, the worst is destroyed. One generation is considered to be the same number of matings as population size."""
    #Create a population of 50 networks
    #population = [build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes)) for each in xrange(int(population_size))]
    population = [mutate_biased(net, 1.0, random_range) for each in xrange(int(population_size) - 1)]
    population.append(net)

    #Rank them
    error = {} #reset errors
    top_networks = [] #reset top five
    #For all networks, simulate, measure their error, and save the best network so far
    #sim_results = mp_nets_sim(population, input_array)
    sim_results = [net.sim(input_array) for net in population]
    for member, sim_result in zip(population, sim_results):
        error[member] = error_function(output_array, sim_result) / len(output_array)
        #compare with rest of population
        for i in xrange(0, max(1, len(top_networks))): #Make sure it enters
            if not top_networks or top_networks[i] and error[member] < error[top_networks[i]]:
                top_networks.insert(i, member)
                break
            elif i == len(top_networks) - 1:
                #Last one, append
                top_networks.append(member)

    #For each generation
    for generation in xrange(int(epochs)):
        try: # I want to catch keyboard interrupt if the user wants to cancel training
            #Select two networks, mate them to create a new network. Repeat population-times.
            for child_index in xrange(len(top_networks)):
                [mother, father] = sample(top_networks[0:int(population_size * 0.65)], 2)

                child = crossover_node(mother, father)

                #Compare errors, and dispose of the worst
                cerror = error_function(output_array, child.sim(input_array)) / len(output_array)
                #compare with rest of population
                for i in xrange(0, len(top_networks)):
                    if top_networks[i] and cerror < error[top_networks[i]]:
                        top_networks.insert(i, child)
                        error[child] = cerror #Add to error dict as well
                        error.pop(top_networks.pop()) #Removes last from both lists
                        break #If child was worse than all, then nothing to be done.

                glogger.debugPlot('Genetic breeding results for training set\nCrossover Green, Mutation red', cerror, style = 'g-', subset = 'crossover')

                parent = sample(top_networks[0:int(population_size * 0.65)], 1)[0]

                child = mutate_biased(parent, mutation_chance, random_range)

                #Compare errors, and dispose of the worst
                cerror = error_function(output_array, child.sim(input_array)) / len(output_array)
                #compare with rest of population
                for i in xrange(0, len(top_networks)):
                    if top_networks[i] and cerror < error[top_networks[i]]:
                        top_networks.insert(i, child)
                        error[child] = cerror #Add to error dict as well
                        error.pop(top_networks.pop()) #Removes last from both lists
                        break #If child was worse than all, then nothing to be done.

                glogger.debugPlot('Genetic breeding results for training set\nCrossover Green, Mutation red', cerror, style = 'r-', subset = 'mutation')

            if validation_inputs is not None and validation_targets is not None:
                verror = error_function(validation_targets, top_networks[0].sim(validation_inputs)) / len(validation_targets)
            else:
                verror = 0
            logger.info("Generation " + str(generation) + ", best: " + str(error[top_networks[0]]) + " validation: " + str(verror))
            glogger.debugPlot('Test Error\nTraining Green, Validation red\nSize: ' + str(len(top_networks[0].hidden_nodes)), error[top_networks[0]], style = 'g-', subset = 'training')
            glogger.debugPlot('Test Error\nTraining Green, Validation red\nSize: ' + str(len(top_networks[0].hidden_nodes)), verror, style = 'r-', subset = 'validation')
        except KeyboardInterrupt:
            logger.info("Interrupt received, returning best net so far...")
            break

    #finally, return the best network
    return top_networks[0]
