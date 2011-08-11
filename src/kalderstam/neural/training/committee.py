from kalderstam.neural.mp_network import mp_train_committee

def train_committee(com, train_func, input_array, target_array, *train_args, **train_kwargs):
    '''Returns error on test set, validation set. Saves network "inplace".'''
    return mp_train_committee(com, train_func, input_array, target_array, *train_args, **train_kwargs)
