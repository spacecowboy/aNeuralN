import os
import numpy
from multiprocessing import cpu_count, Pool

def __split_inputs(inputs, number_of_pieces):
    if len(inputs) <= 2*number_of_pieces:
        return inputs
    else:
        return numpy.array_split(inputs, number_of_pieces)

def net_sim((net, args, kwargs)):
    print('process id:' + str(os.getpid()) + " Starting...")
    #Y = net.sim(*args, **kwargs)
    Y = 1
    print('process id:' + str(os.getpid()) + " Done!")
    return Y

def mp_net_sim_inputs(net, inputs):
    cmd_list = [(net, [input_part], {}) for input_part in __split_inputs(inputs, cpu_count())]
    results = p.map(net_sim, cmd_list)
    print results

def mp_nets_sim(nets, inputs):
    pass

p = Pool()

if __name__ == '__main__':
    mp_net_sim_inputs("net", numpy.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0]))
