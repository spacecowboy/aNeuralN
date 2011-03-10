import os
import numpy
from multiprocessing import cpu_count, Pool
from kalderstam.neural.network import build_feedforward
from kalderstam.neural.matlab_functions import loadsyn1
from kalderstam.util.decorators import benchmark

def __split_inputs(inputs, number_of_pieces):
    if len(inputs) <= 2*number_of_pieces:
        return inputs
    else:
        return numpy.array_split(inputs, number_of_pieces)

def net_sim((net, args, kwargs)):
    #print('process id:' + str(os.getpid()) + " Starting...")
    Y = net.sim(*args, **kwargs)
    #print('process id:' + str(os.getpid()) + " Done!")
    return Y

def mp_net_sim_inputs(net, inputs):
    """Splits the input into cpu_count pieces, and evaluates the network on each piece in a separate process."""
    cmd_list = [(net, [input_part], {}) for input_part in __split_inputs(inputs, cpu_count())]
    pre_results = p.map(net_sim, cmd_list)
    
    results = []
    for result_set in pre_results:
        for result in result_set:
            results.append(result)
    
    return numpy.array(results)

def mp_nets_sim(nets, inputs):
    pass

p = Pool()

if __name__ == '__main__':
    net = build_feedforward(2, 1, 1)
    
    P, T = loadsyn1(250)
    
    Y = benchmark(mp_net_sim_inputs)(net, P)
    
    benchmark(net.sim)(P)
    #print Y
    print len(Y)
