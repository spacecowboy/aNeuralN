from kalderstam.neural.mp_network import mp_train_committee

def train_committee(com, train_func, input_array, target_array, *train_args, **train_kwargs):
    return mp_train_committee(com, train_func, input_array, target_array, *train_args, **train_kwargs)
