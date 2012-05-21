import logging
import numpy
from random import sample, random, choice
from ann.network import node, network
from ann.errorfunctions import sumsquare_total
#import kalderstam.util.graphlogger as glogger


logger = logging.getLogger('ann.trainingfunctions')

numpy.seterr(all = 'raise') #I want errors!

def get_limited_range(mutation_range, half_point, epoch):
    '''
    initial mutation_range, desired half_point, current epoch
    Returns a limited mutation range specified by the logsig function.
    The user specifies where he wants range to assume half of the initial value.
    If half_point = total_number_of_epochs / 2, then it will basically zero close to the end of training.
    Less will make it decrease faster, bigger will make it decrease slower.
    '''
    return mutation_range / (1 + numpy.exp(epoch - half_point))

def select_parents(mutation_chance, pop_size):
    '''
    Using the geometric distrubution, select two networks to mate.
    returns the indices of the networks. Greatest probability to select network 0.
    '''

    mother, father = -1, -1
    while mother < 0 or mother >= pop_size:
        mother = numpy.random.geometric(mutation_chance) - 1 #Because 1 is the smallest that will be returned
    while father < 0 or father >= pop_size or father == mother:
        father = numpy.random.geometric(mutation_chance) - 1 #Because 1 is the smallest that will be returned

    return mother, father

def select_parent(mutation_chance, pop_size):
    '''
    Using the geometric distrubution, select one network to mate.
    returns the indices of the network. Greatest probability to select network 0.
    '''

    mother = -1
    while mother < 0 or mother >= pop_size:
        mother = numpy.random.geometric(mutation_chance) - 1 #Because 1 is the smallest that will be returned

    return mother


def crossover_node(mother, father):
    child = network()
    child.num_of_inputs = mother.num_of_inputs

    tr = {}
    for i in xrange(mother.num_of_inputs):
        tr[i] = i

    for bias in (mother.bias_node, father.bias_node):
        tr[bias] = child.bias_node

    for mother_node, father_node in zip(mother.hidden_nodes, father.hidden_nodes):
        child_node = node(mother_node.activation_function, 1)
        tr[mother_node] = child_node
        tr[father_node] = child_node

        #choose one node to pass on
        _choice = sample([mother_node, father_node], 1)[0]

        #map the weights and bias of that node to the child node
        for keynode, weight in _choice.weights.items():
            child_node.weights[tr[keynode]] = weight

        child.hidden_nodes.append(child_node)

    for mother_node, father_node in zip(mother.output_nodes, father.output_nodes):
        child_node = node(mother_node.activation_function, 1)
        tr[mother_node] = child_node
        tr[father_node] = child_node

        #choose one node to pass on
        _choice = sample([mother_node, father_node], 1)[0]

        #map the weights and bias of that node to the child node
        for keynode, weight in _choice.weights.items():
            child_node.weights[tr[keynode]] = weight

        child.output_nodes.append(child_node)

    return child

def mutate_biased(parent, random_mean, mutation_chance = 0.1):
    '''
    Random mean decides the mean of the exponential distribution where weight
    adjustments are chosen from.
    '''
    child = crossover_node(parent, parent)

    mutate_biased_inplace(child, random_mean, mutation_chance)

    return child

def mutate_biased_inplace(child, random_mean, mutation_chance = 0.1):
    '''
    Random mean decides the mean of the exponential distribution where weight
    adjustments are chosen from.
    '''
    for node in child.get_all_nodes():
        for keynode, weight in node.weights.items():
            if (random() < mutation_chance): # mutation chance
                #node.weights[keynode] += uniform(-random_range, random_range)
                node.weights[keynode] += choice([-1, 1]) * numpy.random.exponential(random_mean)
                #if (random() < 0.25): # even less chance of this happening
                    # Reverse the sign
                #    node.weights[keynode] *= -1

def train_davis_genetic(net, (input_array, output_array), (validation_inputs, validation_targets), epochs = 300, population_size = 50.0, mutation_chance = 0.05, random_mean = 0.5, mutation_half_point = 499, error_function = sumsquare_total, loglevel = None, *args, **kwargs): #@UnusedVariable
    """Creates more networks and evolves the best it can.
    Uses validation set only for plotting.
    This version does not replace the entire population each generation. Two parents are selected at random to create a child.
    This child is inserted into the list sorted on error. The worst network is destroyed if population exceeds limit.
    One generation is considered to be the same number of matings as population size.
    Networks to be mated are selected with the geometric distribution, probability of the top network to be chosen = 0.05
    Mutation chance dictate the probability of every single weight being mutated."""
    if loglevel is not None:
        logging.basicConfig(level = loglevel)
    #Create a population of 50 networks
    #population = [build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes)) for each in xrange(int(population_size))]
    population = [mutate_biased(net, random_mean, mutation_chance = 1.0) for each in xrange(int(population_size) - 1)]
    population.append(net)

    #Rank them
    error = {} #reset errors
    sorted_errors = []
    top_networks = [] #reset top list
    #For all networks, simulate, measure their error, and save the best network so far
    #sim_results = mp_nets_sim(population, input_array)
    sim_results = [net.sim(input_array) for net in population]
    for member, sim_result in zip(population, sim_results):
        error[member] = error_function(output_array, sim_result) / len(output_array)
        #compare with rest of population
        for i in xrange(0, max(1, len(top_networks))): #Make sure it enters
            if not top_networks or top_networks[i] and error[member] < sorted_errors[i]:
                top_networks.insert(i, member)
                sorted_errors.insert(i, error[member])
                break
            elif i == len(top_networks) - 1:
                #Last one, append
                top_networks.append(member)
                sorted_errors.append(error[member])
    logger.debug(sorted_errors)

    def insert_child(child):
        #Compare errors, and dispose of the worst
        cerror = error_function(output_array, child.sim(input_array)) / len(output_array)
        #compare with rest of population
        for i in xrange(0, len(top_networks)):
            if top_networks[i] and cerror < sorted_errors[i]:
                top_networks.insert(i, child)
                sorted_errors.insert(i, cerror)
                error[child] = cerror #Add to error dict as well

                if len(top_networks) > population_size:
                    error.pop(top_networks.pop()) #Removes last from both lists
                    sorted_errors.pop()
                break
        if cerror >= sorted_errors[-1] and len(top_networks) < population_size:
            #belongs last in the list
            top_networks.append(child)
            sorted_errors.append(cerror)
            error[child] = cerror #Add to error dict as well
            i = -1

        return i

    def remove_child(index = None, child = None):
        if index is None:
            index = top_networks.index(child)
        child = top_networks.pop(index)
        error.pop(child)
        sorted_errors.pop(index)
        return child

    #For each generation
    for generation in xrange(int(epochs)):
        try: # I want to catch keyboard interrupt if the user wants to cancel training
            #Select two networks, mate them to create a new network. Repeat population-times.
            for child_index in xrange(len(top_networks)):

                #First breed a new child, with slight defects. These will displace all others with time
                parents = select_parents(0.05, population_size)
                mother, father = top_networks[parents[0]], top_networks[parents[1]]

                child = crossover_node(mother, father)

                mutate_biased_inplace(child, get_limited_range(random_mean, mutation_half_point, generation), mutation_chance)

                insert_child(child)

                #Just mutation
                #parent = top_networks[select_parent(0.05, population_size)]
                #child = mutate_biased(parent, random_mean, mutation_chance)
                #insert_child(child)

                #To prevent this, we keep the lower half randomized.

                #i = int(population_size / 2) + select_parent(0.05, int(population_size / 2))

                #mutant = remove_child(i)
                #mutate_biased_inplace(child, random_mean, mutation_chance = 1.0)

                #insert_child(mutant)

                #insert_child(child)

                #glogger.debugPlot('Genetic breeding results for training set\nCrossover Green, Mutation red', sorted_errors[i], style = 'g-', subset = 'crossover')

                #Completely random selection
                #mutant = sample(top_networks, 1)[0]
                #remove_child(child = mutant)

                #Greater chance to select top networks
                #i = select_parent(0.05, population_size)

                #child = remove_child(index = i)
                #mutate_biased_inplace(child, random_mean, mutation_chance)

                #child = mutate_biased(top_networks[i], random_mean, mutation_chance)

                #i = insert_child(child)

                #glogger.debugPlot('Genetic breeding results for training set\nCrossover Green, Mutation red', sorted_errors[i], style = 'r-', subset = 'mutation')

            if validation_inputs is not None and validation_targets is not None and \
                len(validation_inputs) > 0 and len(validation_targets) > 0:
                verror = error_function(validation_targets, top_networks[0].sim(validation_inputs)) / len(validation_targets)
            else:
                verror = 0
            logger.info("Generation " + str(generation) + ", best: " + str(error[top_networks[0]]) + " validation: " + str(verror))
            #glogger.debugPlot('Test Error\nTraining Green, Validation red\nSize: ' + str(len(top_networks[0].hidden_nodes)), error[top_networks[0]], style = 'g-', subset = 'training')
            #glogger.debugPlot('Test Error\nTraining Green, Validation red\nSize: ' + str(len(top_networks[0].hidden_nodes)), verror, style = 'r-', subset = 'validation')
        except KeyboardInterrupt:
            logger.info("Interrupt received, returning best net so far...")
            break

    #finally, return the best network
    logger.debug(sorted_errors)
    return top_networks[0]
