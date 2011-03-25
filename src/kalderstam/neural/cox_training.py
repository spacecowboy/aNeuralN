from kalderstam.neural.error_functions.cox_error import derivative, calc_beta,\
    calc_sigma, get_risk_outputs
    
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

def train_cox(net, inputs, timeslots, epochs = 300, learning_rate = 1):
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        outputs = net.sim(inputs)
        #Check if beta will diverge here, if so, end training with error 0
        if beta_diverges(outputs, timeslots):
            #End training
            break
        beta = calc_beta(outputs, timeslots)
        sigma = calc_sigma(outputs)
            
        #Set corrections to 0 on all nodes first
        for node in net.get_all_nodes():
            node.weight_corrections = {}
        
        #Iterate over all output indices
        for input, output_index in zip(inputs, range(len(outputs))):
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

if __name__ == '__main__':
    from kalderstam.neural.matlab_functions import loadsyn1, stat, plot2d2c, \
    loadsyn2, loadsyn3, plotroc
    from kalderstam.util.filehandling import parse_file, save_network
    from kalderstam.util.decorators import benchmark
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    from random import uniform
    import time
    import numpy
    import matplotlib.pyplot as plt
    
    #the function the network should try and approximate
    def sickness_sim(x):
        return x[0]*3
    
    #the values the network have to go on
    def sickness_with_noise(x, noise_level = 1):
        actual_values = sickness_sim(x)
        #Add some noise
        return actual_values + noise_level*uniform(-1, 1)
    
    def generate_timeslots(x_array):
        timeslots = numpy.array([], dtype = int)
        for x_index in range(len(x_array)):
            x = x_array[x_index]
            if len(timeslots) == 0:
                timeslots = numpy.insert(timeslots, 0, x_index)
            else:
                added = False
                #Find slot
                for time_index in timeslots:
                    if sickness_with_noise(x, noise_level=2) > sickness_with_noise(x_array[time_index], noise_level=2):
                        timeslots = numpy.insert(timeslots, time_index, x_index)
                        added = True
                        break
                if not added:
                    #Reached the end, insert here
                    timeslots = numpy.append(timeslots, x_index)
                
        return timeslots
    
    p = 2 #number of input covariates
        
    net = build_feedforward(p, 2, 1)
    
    num_of_patients = 50
    x_array = numpy.array([[uniform(0, 10), uniform(0, 10)] for i in range(num_of_patients)])
    print x_array
    
    real_targets = [sickness_sim(x) for x in x_array]
    print real_targets
    noise_targets = [sickness_with_noise(x, noise_level=3) for x in x_array]
    print noise_targets
    
    timeslots = generate_timeslots(x_array)
    print timeslots
    
    output_before_training = net.sim(x_array)
    print "output_before_training"
    print output_before_training
    
    net = train_cox(net, x_array, timeslots, epochs = 10, learning_rate = 1)
    
    output_after_training = net.sim(x_array)
    print "output_after_training"
    print output_after_training