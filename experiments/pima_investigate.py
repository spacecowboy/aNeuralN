from os import path
from kalderstam.util.filehandling import parse_file, save_committee,\
    load_committee
from kalderstam.neural.network import build_feedforward_committee,\
    build_feedforward
from kalderstam.util.decorators import benchmark
from kalderstam.neural.training_functions import train_committee,\
    traingd_block, train_evolutionary
import logging
from kalderstam.neural.matlab_functions import plotroc, stat
import matplotlib.pyplot as plt
    
logging.basicConfig(level = logging.DEBUG)
#load the training set
filename = path.join(path.expanduser("~"), "Kurser/ann_FYTN06/exercise1/pima_trn.dat")
inputs, targets = parse_file(filename, targetcols = 8)

#load the test set
filename = path.join(path.expanduser("~"), "Kurser/ann_FYTN06/exercise1/pima_tst.dat")
test_inputs, tst_t = parse_file(filename)



test = (inputs, targets)
validation = ([], [])

com = load_committee("/export/home/jonask/Projects/aNeuralN/ANNs/pimatrain_rocarea84.0328358209.anncom")

#Estimate on test set now
#Y_test = com.sim(test_inputs)
#for value in Y_test:
#    print value[0]
    
Y_neg = com.update(test_inputs[68])
print Y_neg