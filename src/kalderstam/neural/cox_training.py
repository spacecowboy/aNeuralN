from kalderstam.neural.error_functions.cox_error import derivative, calc_beta,\
    calc_sigma

def train_cox(net, inputs, timeslots, epochs = 300, learning_rate = 1):
    for epoch in epochs:
        outputs = net.sim(inputs)
        beta = calc_beta(outputs)
        sigma = calc_sigma(outputs)
            
        #Set corrections to 0 on all nodes first
        for node in net.get_all_nodes():
            node.weight_corrections = {}
        
        #Iterate over all output indices
        for output_index in range(len(outputs)):
            #Set error to 0 on all nodes first
            for node in net.get_all_nodes():
                node.error_gradient = 0

            #Set errors on output nodes first
            for node, gradient in zip(net.output_nodes, derivative(beta, sigma, output_index, outputs, timeslots)):
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
        
    return net
