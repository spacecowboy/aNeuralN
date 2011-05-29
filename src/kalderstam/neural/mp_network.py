import numpy
from multiprocessing import cpu_count, Pool
from kalderstam.util.filehandling import get_stratified_validation_set,\
    get_validation_set
import pickle

def __split_inputs(inputs, number_of_pieces):
    if len(inputs) <= 2*number_of_pieces:
        return inputs
    else:
        return numpy.array_split(inputs, number_of_pieces)

def __net_sim((net, args, kwargs)):
    #print('process id:' + str(os.getpid()) + " Starting...")
    print("mam")
    Y = net.sim(*args, **kwargs)
    print("mom")
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
    print("fam", nets)
    cmd_list = [(net, [inputs], {}) for net in nets]
    print("fom")
    results = p.map(__net_sim, cmd_list)
    print("fim")
    return results

def mp_committee_sim(com, inputs):
    """This evaluates each network in a separate process. Does not split the input.
    The results are returned in the same order as the iterator of nets"""
    results = mp_nets_sim(com.nets, inputs)
    return com.__average__(results)

def mp_train_committee(com, train_func, input_array, target_array, *train_args, **train_kwargs):
    #Do stratified or not?
    do_strat = True
    try:
        if len(target_array[0]) > 1:
            do_strat = False
    except TypeError: #In this case, it's an array of single values. we should do strat
        pass #Already true
    if do_strat:
        data_sets = [get_stratified_validation_set(input_array, target_array, validation_size = 1/len(com)) for times in range(len(com))]
    else:
        data_sets = [get_validation_set(input_array, target_array, validation_size = 1/len(com)) for times in range(len(com))]

    cmd_list = [(net, train_func, T, V, train_args, train_kwargs) for net, (T, V) in zip(com.nets, data_sets)]
    trained_nets = p.map(__train_net, cmd_list)
    com.nets = trained_nets
    
def __train_net((net, train_func, T, V, train_args, train_kwargs)):
    return train_func(net, T, V, *train_args, **train_kwargs)
    

p = Pool()

if __name__ == '__main__':
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    from kalderstam.matlab.matlab_functions import loadsyn1
    from kalderstam.util.decorators import benchmark

    net = build_feedforward(2, 4, 1)
    
    P, T = loadsyn1(10000)
    
    print("fire")
    Y = benchmark(mp_net_sim_inputs)(net, P)
    print("fire done")
    
    benchmark(net.sim)(P)
    #print Y
    print len(Y)
    print len(Y[0])
    
    netlist = [build_feedforward(2, 4, 1) for i in range(4)]
    print("bim")
    Ys = benchmark(mp_nets_sim)(netlist, P)
    print("bam")
    
    def single_nets(nets, P):
        for net in nets:
            net.sim(P)
    print("sho")
    benchmark(single_nets)(netlist, P)
    
    print len(Ys)
    print len(Ys[0])
    print len(Ys[0][0])
    
    com = build_feedforward_committee(input_number = 2, hidden_number = 4, output_number = 1)
    print("built")
    Ys = benchmark(mp_committee_sim)(com, P)
    benchmark(com.sim)(P)
    print len(Ys)
    print len(Ys[0])
    print "done"