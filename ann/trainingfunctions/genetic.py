import logging
import numpy
from random import sample, random, choice
from ann.network import build_feedforward, node, network, \
    connect_nodes
from ann.errorfunctions import sumsquare_total
#import kalderstam.util.graphlogger as glogger

logger = logging.getLogger('kalderstam.neural.training_functions')

numpy.seterr(all = 'raise') #I want errors!

def mutate_biased_inplace(child, random_mean = 0.2, mutation_chance = 0.25):
    '''
    Random mean decides the mean of the exponential distribution where weight
    adjustments are chosen from.
    '''
    for _node in child.get_all_nodes():
        for keynode, weight in _node.weights.iteritems():
            if (random() < mutation_chance): # mutation chance
                #node.weights[keynode] += uniform(-random_range, random_range)
                _node.weights[keynode] += choice([-1, 1]) * numpy.random.exponential(random_mean)

def train_evolutionary(net, (input_array, output_array), (validation_inputs, validation_targets), epochs = 1, population_size = 100, mutation_chance = 0.25, random_range = 1.0, random_mean = 0.2, top_number = 25, cross_over_chance = 0.5, error_function = sumsquare_total, loglevel = None, *args, **kwargs): #@UnusedVariable
    """Creates more networks and evolves the best it can.
    Does NOT use any validation set..."""
    if top_number > population_size:
        raise ValueError('Top_number({0}) can not be larger than population size({1})!'.format(top_number, population_size))

    if loglevel is not None:
        logging.basicConfig(level = loglevel)
    #Create a population of 50 networks
    best = None
    best_error = None
    best_val_error = None
    population = [build_feedforward(net.num_of_inputs, len(net.hidden_nodes), len(net.output_nodes)) for each in xrange(int(population_size))]
    #For each generation
    for generation in xrange(int(epochs)):
        try: # I want to catch keyboard interrupt if the user wants to cancel training
            error = {} #reset errors
            top_networks = [None for each in xrange(int(top_number))] #reset top five
            #For all networks, simulate, measure their error, and save the best network so far
            #sim_results = mp_nets_sim(population, input_array)
            sim_results = [net.sim(input_array) for net in population]
            for member, sim_result in zip(population, sim_results):
                error[member] = error_function(output_array, sim_result) / len(output_array)
                #compare with best
                if not best or error[member] < best_error:
                    best = member
                    best_error = error[member]
                    if validation_inputs is not None and len(validation_inputs) > 0:
                        val_sim_results = best.sim(validation_inputs)
                        best_val_error = error_function(validation_targets, val_sim_results) / len(validation_targets)
                    else:
                        best_val_error = -1
                #compare with five best this generation
                comp_net = member
                for i in xrange(0, len(top_networks)):
                    if top_networks[i] and error[comp_net] < error[top_networks[i]]:
                        old = top_networks[i]
                        top_networks[i] = comp_net
                        comp_net = old
                    elif not top_networks[i]:
                        top_networks[i] = comp_net
                        break

            logger.info("Generation {0}, best trn: {1} val: {2}, top trn: {3}".format(generation, best_error, best_val_error, error[top_networks[0]]))
            #glogger.debugPlot('Test Error\nMutation rate: ' + str(mutation_chance), best_error, style = 'r-', subset='best')
            #glogger.debugPlot('Test Error\nMutation rate: ' + str(mutation_chance), error[top_networks[0]], style = 'b-', subset='top')

            #Select the best 5 networks, mate them randomly and create 50 new networks
            population = []
            for child_index in xrange(population_size):
                [mother, father] = sample(top_networks, 2)
                if random() < cross_over_chance:
                    father = mother #Will create full mutation, no cross-over

                population.append(network())
                population[child_index].num_of_inputs = mother.num_of_inputs
                bias_child = population[child_index].bias_node
                #Hidden layer
                for mother_node, father_node in zip(mother.hidden_nodes, father.hidden_nodes):
                    hidden = node(mother_node.activation_function, random_range)
                    weights = {}
                    for input_number in xrange(mother.num_of_inputs):
                        choice = sample([mother_node, father_node], 1)[0]
                        weights[input_number] = choice.weights[input_number]
                    choice = sample([(mother, mother_node), (father, father_node)], 1)[0]
                    weights[bias_child] = choice[1].weights[choice[0].bias_node]
                    
                    _all = range(mother.num_of_inputs)
                    _all.append(bias_child)
                    connect_nodes(hidden, _all, weights)
                    population[child_index].hidden_nodes.append(hidden)

                hidden_nodes = population[child_index].hidden_nodes

                #Output nodes
                #for mother_node, father_node in zip(mother.output_nodes, father.output_nodes):
                for output_number in xrange(len(mother.output_nodes)):
                    output = node(mother.output_nodes[output_number].activation_function)
                    weights = {}
                    for hidden_number in xrange(len(mother.hidden_nodes)):
                        choice = sample([mother, father], 1)[0]
                        weights[hidden_nodes[hidden_number]] = choice.output_nodes[output_number].weights[choice.hidden_nodes[hidden_number]]
                    choice = sample([mother, father], 1)[0]
                    weights[bias_child] = choice.output_nodes[output_number].weights[choice.bias_node]

                    _all = [] + hidden_nodes
                    _all.append(bias_child)
                    connect_nodes(output, _all, weights)

                    population[child_index].output_nodes.append(output)

                #Mutate it
                mutate_biased_inplace(population[child_index], random_mean, mutation_chance)

        except KeyboardInterrupt:
            logger.info("Interrupt received, returning best net so far...")
            break

    logger.debug("Error for top networks: " + str([error[mem] for mem in top_networks]))
    #finally, return the best network
    return best
