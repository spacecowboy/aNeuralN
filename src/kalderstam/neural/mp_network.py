import numpy
from multiprocessing import cpu_count, Pool
from kalderstam.util.filehandling import get_cross_validation_sets
import pickle
import logging
logger = logging.getLogger('kalderstam.neural.network.mp_trainer')

def __split_inputs(inputs, number_of_pieces):
    if len(inputs) <= 2 * number_of_pieces:
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

def mp_train_committee(com, train_func, input_array, target_array, binary_target = None, val_num = 3, *train_args, **train_kwargs):
    '''Returns error on test set, validation set. Saves network "inplace".
    binary_target is the column number in the target_array which is binary, and should be used for getting a 
    stratified validation set (proportional number of ones and zeros in the validation set and training set).
    val_num is the number of pieces to divide the data set into for cross validation training. Default is 3 and that will create
    a validation set of size 1/3. This also has the effect that the length of the committee must be a multiple of val_num or an exception
    is thrown. Because what is done is a IxK-crossfold validation. Where K i s val_num and I is com_length / K.'''

    if len(com) % val_num != 0:
        raise ValueError('Committee length is not a multiple of val_num: ' + str(val_num))
    if val_num < 2:
        raise ValueError('Val_num = 1. Training a committee on the exact same data is useless!')

    I = len(com) / val_num

    #If the committee is one in length, still divide in two pieces, but only keep one of them.
    #if len(com) == 1:
    #    data_sets = [get_cross_validation_sets(input_array, target_array, 2 , binary_column = binary_target)[0]]
    #else:
    #    data_sets = get_cross_validation_sets(input_array, target_array, len(com) , binary_column = binary_target)

    #Construct appropriate data_sets for training
    data_sets = []
    set_indices = []
    for part in xrange(I):
        #Add these datasets to the list
        (data, indices) = get_cross_validation_sets(input_array, target_array, val_num, binary_column = binary_target, return_indices = True)
        data_sets.extend(data)
        set_indices.extend(indices)

    debug_level = logger.getEffectiveLevel()
    train_kwargs['loglevel'] = debug_level

    cmd_list = [(net, train_func, T, V, train_args, train_kwargs) for net, (T, V) in zip(com.nets, data_sets)]
    trained_nets = p.map(__train_net, cmd_list)
    com.nets = trained_nets

    #Get  errors
    test_errors = {}
    vald_errors = {}

    for net, (T, V) in zip(com.nets, data_sets):
        outputs = net.sim(T[0])
        test_errors[net] = train_kwargs['error_function'](T[1], outputs) / len(outputs)

        outputs = net.sim(V[0])
        vald_errors[net] = train_kwargs['error_function'](V[1], outputs) / len(outputs)

    return com, test_errors, vald_errors, data_sets, set_indices

def __train_net((net, train_func, T, V, train_args, train_kwargs)):
    return train_func(net, T, V, *train_args, **train_kwargs)

num = min(cpu_count() - 1, 3)

p = Pool(num)

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
    print(len(Y))
    print(len(Y[0]))

    netlist = [build_feedforward(2, 4, 1) for i in xrange(4)]
    print("bim")
    Ys = benchmark(mp_nets_sim)(netlist, P)
    print("bam")

    def single_nets(nets, P):
        for net in nets:
            net.sim(P)
    print("sho")
    benchmark(single_nets)(netlist, P)

    print(len(Ys))
    print(len(Ys[0]))
    print(len(Ys[0][0]))

    com = build_feedforward_committee(input_number = 2, hidden_number = 4, output_number = 1)
    print("built")
    Ys = benchmark(mp_committee_sim)(com, P)
    benchmark(com.sim)(P)
    print(len(Ys))
    print(len(Ys[0]))
    print("done")
