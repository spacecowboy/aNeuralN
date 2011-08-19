from kalderstam.neural.mp_network import mp_train_committee

def train_committee(com, train_func, input_array, target_array, binary_target = None, *train_args, **train_kwargs):
    '''Returns error on test set, validation set. Saves network "inplace".
    binary_target is the column number in the target_array which is binary, and should be used for getting a 
    stratified validation set (proportional number of ones and zeros in the validation set and training set).'''
    return mp_train_committee(com, train_func, input_array, target_array, binary_target, *train_args, **train_kwargs)
