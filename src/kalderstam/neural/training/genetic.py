import logging
import numpy
from random import sample, random, uniform
from kalderstam.neural.network import build_feedforward, node, network, \
    connect_nodes
from kalderstam.neural.error_functions import sum_squares
import kalderstam.util.graphlogger as glogger

logger = logging.getLogger('kalderstam.neural.training_functions')

numpy.seterr(all = 'raise') #I want errors!

def train_evolutionary(net, (input_array, output_array), (validation_inputs, validation_targets), epochs = 300, population_size = 50, mutation_chance = 0.05, random_range = 1, top_number = 5, error_function = sum_squares.total_error, *args): #@UnusedVariable
    """Creates more networks and evolves the best it can.
    Does NOT use any validation set..."""
    #Create a population of 50 networks
    best = None
    best_error = None
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

            #Select the best 5 networks, mate them randomly and create 50 new networks
            for child_index in xrange(len(population)):
                [mother, father] = sample(top_networks, 2)

                population[child_index] = network()
                population[child_index].num_of_inputs = mother.num_of_inputs

                #Hidden layer
                for mother_node, father_node in zip(mother.hidden_nodes, father.hidden_nodes):
                    hidden = node(mother_node.activation_function, random_range)
                    weights = {}
                    for input_number in xrange(mother.num_of_inputs):
                        choice = sample([mother_node, father_node], 1)[0]
                        weights[input_number] = choice.weights[input_number]
                        if (random() < mutation_chance): # mutation chance
                            weights[input_number] += uniform(-random_range, random_range)
                    #Don't forget bias
                    if (random() < mutation_chance): # mutation chance
                        hidden.bias = uniform(-random_range, random_range)
                    else:
                        hidden.bias = sample([mother_node, father_node], 1)[0].bias

                    connect_nodes(hidden, xrange(mother.num_of_inputs), weights)
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
                        if (random() < mutation_chance): # mutation chance
                            weights[hidden_nodes[hidden_number]] += uniform(-random_range, random_range)
                    #Don't forget bias
                    if (random() < mutation_chance): # mutation chance
                        output.bias = uniform(-random_range, random_range)
                    else:
                        output.bias = sample([mother, father], 1)[0].output_nodes[output_number].bias

                    connect_nodes(output, hidden_nodes, weights)
                    population[child_index].output_nodes.append(output)

            logger.info("Generation " + str(generation) + ", best so far: " + str(best_error))
            glogger.debugPlot('Test Error', best_error, style = 'r-')
        except KeyboardInterrupt:
            logger.info("Interrupt received, returning best net so far...")
            break

    #finally, return the best network
    return best
