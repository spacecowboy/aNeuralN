from kalderstam.neural.error_functions.cox_error import derivative, calc_beta,\
    calc_sigma, get_risk_outputs, total_error
from kalderstam.util.decorators import benchmark
import logging

logger = logging.getLogger('kalderstam.neural.cox_training')
    
def beta_diverges(outputs, timeslots):
    diverging = True
    diverging_negatively = True
    #Check every timeslot and thus every riskgroup
    for s in timeslots:
        risk_outputs = get_risk_outputs(s, timeslots, outputs)
        for risk in risk_outputs[1:]: #First one will always be s
            if outputs[s] < risk: #It's not the largest, no risk for positive divergence
                diverging = False
            if outputs[s] > risk: #It's not the smallest, no risk for negative divergence
                diverging_negatively = False
    return (diverging or diverging_negatively)

@benchmark
def train_cox(net, inputs, timeslots, epochs = 300, learning_rate = 0.1):
    numpy.seterr(all='raise') #I want errors!
    for epoch in range(epochs):
        logger.debug("Epoch " + str(epoch))
        outputs = net.sim(inputs)
        #Check if beta will diverge here, if so, end training with error 0
        if beta_diverges(outputs, timeslots):
            #End training
            logger.info('Beta diverges...')
            break
        try:
            beta = calc_beta(outputs, timeslots)
        except FloatingPointError as e:
            print(str(e))
            break #Stop training
        sigma = calc_sigma(outputs)
            
        #Set corrections to 0 on all nodes first
        for node in net.get_all_nodes():
            node.weight_corrections = {}
        
        #Iterate over all output indices
        i = 0
        for input, output_index in zip(inputs, range(len(outputs))):
            logger.debug("Patient: " + str(i))
            i += 1
            #Set error to 0 on all nodes first
            for node in net.get_all_nodes():
                node.error_gradient = 0

            #Set errors on output nodes first
            for node, gradient in zip(net.output_nodes, derivative(beta, sigma, output_index, outputs, timeslots)):
                node.error_gradient = gradient
            
            #Iterate over the nodes and correct the weights
            for node in net.output_nodes + net.hidden_nodes:
                #Calculate local error gradient
                node.error_gradient *= node.output_derivative(input)

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
        
    return net

def test():
    from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
    loadsyn2, loadsyn3, plotroc, plot_network_weights
    from kalderstam.util.filehandling import parse_file, save_network, load_network
    from kalderstam.util.decorators import benchmark
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    from random import uniform
    import time
    import numpy
    import matplotlib.pyplot as plt
    from kalderstam.neural.activation_functions import linear
    
    numpy.seterr(all='raise')
    logging.basicConfig(level = logging.DEBUG)
    
    #the function the network should try and approximate
    def sickness_sim(x):
        return x[0]*3 + x[1]*6
    
    #the values the network have to go on
    def sickness_with_noise(x, noise_level = 1):
        actual_values = sickness_sim(x)
        #Add some noise
        return actual_values + noise_level*uniform(-1, 1)
    
    def generate_timeslots(P, T):
        timeslots = numpy.array([], dtype = int)
        for x_index in range(len(P)):
            x = P[x_index]
            time = T[x_index][0]
            if len(timeslots) == 0:
                timeslots = numpy.insert(timeslots, 0, x_index)
            else:
                added = False
                #Find slot
                for time_index in timeslots:
                    if time < T[time_index][0]:
                        timeslots = numpy.insert(timeslots, time_index, x_index)
                        added = True
                        break
                if not added:
                    #Reached the end, insert here
                    timeslots = numpy.append(timeslots, x_index)
                
        return timeslots
    
    p = 4 #number of input covariates
        
    #net = build_feedforward(p, 2, 1, output_function = linear())
    #save_network(net, '/home/gibson/jonask/test_net.ann')
    net = load_network('/home/gibson/jonask/test_net.ann')

    P, T = parse_file('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_with_noise.txt', targetcols = [4], inputcols = [0,1,2,3], ignorecols = [], ignorerows = [], normalize = False)
    #P, T = parse_file('/home/gibson/jonask/fake_survival_data_very_small.txt', targetcols = [4], inputcols = [0,1,2,3], ignorecols = [], ignorerows = [], normalize = False)
    #P = P[:100,:]
    #T = T[:100, :]
    print P
    print T
    
    timeslots = generate_timeslots(P, T)
    print timeslots
    
    outputs = net.sim(P)
    print "output_before_training"
    print outputs
    
    beta = calc_beta(outputs, timeslots)
    sigma = calc_sigma(outputs)
    
    plot_network_weights(net, figure=1)
    plt.title('Before training, [hidden, output] vs [input, hidden, output\nError = ' + str(total_error(beta, sigma)))
        
    net = train_cox(net, P, timeslots, epochs = 1, learning_rate = 2)
    outputs = net.sim(P)
    
    plot_network_weights(net, figure=2)
    try:
        beta = calc_beta(outputs, timeslots)
        sigma = calc_sigma(outputs)
        error = total_error(beta, sigma)
    except FloatingPointError:
        error = 'Beta diverged'
    plt.title('After training, [hidden, output] vs [input, hidden, output\nError = ' + str(error))
    
    print "output_after_training"
    print outputs
    
    plt.figure(3)
    plt.title('Scatter plot')
    plt.xlabel('Survival time (with noise) years')
    plt.ylabel('Network output')
    plt.scatter(T.flatten(), outputs.flatten(), c='g', marker='s')

#This is a test of the functionality in this file
if __name__ == '__main__':
    from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
    loadsyn2, loadsyn3, plotroc, plot_network_weights
    from kalderstam.util.filehandling import parse_file, save_network
    from kalderstam.util.decorators import benchmark
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    from random import uniform
    import time
    import numpy
    import matplotlib.pyplot as plt
    from kalderstam.neural.activation_functions import linear
    
    import pstats, cProfile
    
    numpy.seterr(all='raise')
    logging.basicConfig(level = logging.DEBUG)

    cProfile.runctx("test()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    
    plt.show()