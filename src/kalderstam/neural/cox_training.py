from kalderstam.neural.error_functions.cox_error import derivative, calc_beta,\
    calc_sigma, get_risk_outputs, total_error
from kalderstam.util.decorators import benchmark
import logging
from numpy import exp
import numpy as np
import kalderstam.util.graphlogger as glogger

logger = logging.getLogger('kalderstam.neural.cox_training')
betalogger = glogger.getGraphLogger('Beta vs Epochs')
betaforcelogger = glogger.getGraphLogger('BetaForce vs Epochs', 'bs')
sigmalogger = glogger.getGraphLogger('Sigma vs Epochs', 'bs')
    
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

def train_cox(net, (test_inputs, test_targets), (validation_inputs, validation_targets), timeslots, epochs = 1, learning_rate = 2.0):
    numpy.seterr(all='raise') #I want errors!
    inputs = test_inputs
    for epoch in range(epochs):
        logger.info("Epoch " + str(epoch))
        outputs = net.sim(inputs)
        #Check if beta will diverge here, if so, end training with error 0
        #if beta_diverges(outputs, timeslots):
            #End training
        #    logger.info('Beta diverges...')
        #    break
        
        try:
            beta, risk_outputs, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots)
            betalogger.debugplot(beta)
        except FloatingPointError as e:
            print(str(e))
            break #Stop training
        
        #calculate parts of the error function here, which do not depend on patient specific output
        #risk_outputs = [None for i in range(len(timeslots))]
        #beta_risk = [None for i in range(len(timeslots))]
        #part_func = np.zeros(len(timeslots))
        #weighted_avg = np.zeros(len(timeslots))
        beta_force = 0
        #for s in timeslots:
            #risk_outputs[s] = get_risk_outputs(s, timeslots, outputs)  
            #beta_risk[s] = exp(beta*risk_outputs[s])
            #part_func[s] = beta_risk[s].sum()
            #weighted_avg[s] = (beta_risk[s]*risk_outputs[s]).sum()/part_func[s]
            #beta_force += -(beta_risk[s]*risk_outputs[s]**2).sum()/part_func[s] + weighted_avg[s]**2
        beta_force = sum([-(beta_risk[s]*risk_outputs[s]**2).sum()/part_func[s] + weighted_avg[s]**2 for s in timeslots])
        beta_force *= -1
        betaforcelogger.debugplot(beta_force)
        
        sigma = calc_sigma(outputs)
        sigmalogger.debugplot(sigma)
            
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
            for node, gradient in zip(net.output_nodes, derivative(beta, sigma, part_func, weighted_avg, beta_force, output_index, outputs, timeslots)):
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
    #from kalderstam.neural.training_functions import train_committee, traingd_block, train_evolutionary
    
    numpy.seterr(all='raise')
    logging.basicConfig(level = logging.DEBUG)
    glogger.set_logging_level(glogger.debug)
    
    #the function the network should try and approximate
    def sickness_sim(x):
        return 6 - (x[0]*0.25 + x[1]*0.50 + x[2]*0.75 + x[3])
    
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
        
    #net = build_feedforward(p, 4, 1, output_function = linear())
    net = build_feedforward(p, 2, 1, hidden_function = linear(), output_function = linear())
    #save_network(net, '/home/gibson/jonask/test_net.ann')
    #net = load_network('/home/gibson/jonask/test_net.ann')
    
    #com = build_feedforward_committee(size = 4, input_number = p, hidden_number = 6, output_number = 1)
    
    #P, T = parse_file('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_no_noise.txt', targetcols = [4], inputcols = [0,1,2,3], ignorecols = [], ignorerows = [], normalize = False)
    P, T = parse_file('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/new_fake_ann_data_with_noise.txt', targetcols = [4], inputcols = [0,1,2,3], ignorecols = [], ignorerows = [], normalize = False)
    #P, T = parse_file('/home/gibson/jonask/Dropbox/Ann-Survival-Phd/fake_survival_data_with_noise.txt', targetcols = [4], inputcols = [0,1,2,3], ignorecols = [], ignorerows = [], normalize = False)
    #P, T = parse_file('/home/gibson/jonask/fake_survival_data_very_small.txt', targetcols = [4], inputcols = [0,1,2,3], ignorecols = [], ignorerows = [], normalize = False)
    #P = P[:100,:]
    #T = T[:100, :]
    
    timeslots = generate_timeslots(P, T)
    
    outputs = net.sim(P)
    #outputs = com.sim(P)
    #print "output_before_training"
    #print outputs
    
    beta, risk_outputs, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots)
    sigma = calc_sigma(outputs)
    
    #Make beta positive
    #if beta < 0:
    #    for n, w in net.output_nodes[0].weights.iteritems:
    #        net.output_nodes[0].weights[n] = -w
    #    beta, risk_outputs, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots)
    
    plot_network_weights(net)
    #plt.title('Before training, [hidden, output] vs [input, hidden, output\nError = ' + str(total_error(beta, sigma)))
        
    net = train_cox(net, (P, T), (None, None), timeslots, epochs = 200, learning_rate = 2)
    #net = traingd_block(net, (P,T), (None, None), epochs = 25, learning_rate = 1, block_size = 0)
    #net = train_evolutionary(net, (P,T), (None, None), epochs = 500, random_range = 1)
    outputs = net.sim(P)
    
    #train_committee(com, traingd_block, P, T, epochs = 100, block_size = 10)
    #train_committee(com, train_cox, P, T, timeslots, epochs = 10, learning_rate = 2)
    #outputs = com.sim(P)
    
    plot_network_weights(net)
    try:
        beta, risk_outputs, beta_risk, part_func, weighted_avg = calc_beta(outputs, timeslots)
        sigma = calc_sigma(outputs)
        error = total_error(beta, sigma)
    except FloatingPointError:
        error = 'Beta diverged'
    #plt.title('After training, [hidden, output] vs [input, hidden, output\nError = ' + str(error))
    
    #print "output_after_training"
    #print outputs
    
    plt.figure()
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
    logging.basicConfig(level = logging.INFO)
    glogger.set_logging_level(glogger.debug)

    #cProfile.runctx("test()", globals(), locals(), "Profile.prof")

    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()
    
    test()
    plt.show()