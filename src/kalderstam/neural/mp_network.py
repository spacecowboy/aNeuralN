import numpy
from multiprocessing import cpu_count, Pool

def __split_inputs(inputs, number_of_pieces):
    if len(inputs) <= 2*number_of_pieces:
        return inputs
    else:
        return numpy.array_split(inputs, number_of_pieces)

def __net_sim((net, args, kwargs)):
    #print('process id:' + str(os.getpid()) + " Starting...")
    Y = net.sim(*args, **kwargs)
    #print('process id:' + str(os.getpid()) + " Done!")
    return Y

def __mp_net_sim(cmd_list):
    pre_results = p.map(__net_sim, cmd_list)
    
    results = []
    for result_set in pre_results:
        for result in result_set:
            results.append(result)
    
    return numpy.array(results)

def run_on_pool(func, cmd_list):
    return p.map(func, cmd_list)

def mp_net_sim_inputs(net, inputs):
    """Splits the input into cpu_count pieces, and evaluates the network on each piece in a separate process."""
    cmd_list = [(net, [input_part], {}) for input_part in __split_inputs(inputs, cpu_count())]
    pre_results = p.map(__net_sim, cmd_list)
    
    results = []
    for result_set in pre_results:
        for result in result_set:
            results.append(result)
    
    return numpy.array(results)

def mp_nets_sim(nets, inputs):
    """This evaluates each network in a separate process. Does not split the input.
    The results are returned in the same order as the iterator of nets"""
    cmd_list = [(net, [inputs], {}) for net in nets]
    results = p.map(__net_sim, cmd_list)
    
    return results

def mp_committee_sim(com, inputs):
    """This evaluates each network in a separate process. Does not split the input.
    The results are returned in the same order as the iterator of nets"""
    cmd_list = [(net, [inputs], {}) for net in com.nets]
    sim_list = p.map(__net_sim, cmd_list)
    return com.__average__(sim_list)

def mp_train_committee(com, train_func, *train_args, **train_kwargs):
    cmd_list = [(net, train_func, train_args, train_kwargs) for net in com.nets]
    trained_nets = p.map(__train_net, cmd_list)
    com.nets = trained_nets
    
def __train_net((net, train_func, train_args, train_kwargs)):
    return train_func(net, *train_args, **train_kwargs)
    

p = Pool()

if __name__ == '__main__':
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    from kalderstam.neural.matlab_functions import loadsyn1
    from kalderstam.util.decorators import benchmark

    net = build_feedforward(2, 4, 1)
    
    P, T = loadsyn1(10000)
    
    Y = benchmark(mp_net_sim_inputs)(net, P)
    
    benchmark(net.sim)(P)
    #print Y
    print len(Y)
    print len(Y[0])
    
    netlist = [build_feedforward(2, 4, 1) for i in range(4)]
    Ys = benchmark(mp_nets_sim)(netlist, P)
    
    def single_nets(nets, P):
        for net in nets:
            net.sim(P)
    
    benchmark(single_nets)(netlist, P)
    
    print len(Ys)
    print len(Ys[0])
    print len(Ys[0][0])
    
    com = build_feedforward_committee(input_number = 2, hidden_number = 4, output_number = 1)
    Ys = benchmark(mp_committee_sim)(com, P)
    benchmark(com.sim)(P)
    print len(Ys)
    print len(Ys[0])