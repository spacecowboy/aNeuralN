from os import path
from kalderstam.util.filehandling import parse_file
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

#Train!

net = build_feedforward(8, 6, 1)

epochs = 1000

print "Training..."
#best = benchmark(train_evolutionary)(net, test, validation, epochs/10, random_range = 5)
#best = benchmark(traingd_block)(net, test, validation, epochs, block_size = 10, stop_error_value = 0)

com = build_feedforward_committee(size = 10, input_number = 8, hidden_number = 8, output_number = 1)

benchmark(train_committee)(com, train_evolutionary, inputs, targets, 25, random_range = 5)

benchmark(train_committee)(com, traingd_block, inputs, targets, epochs, block_size = 10, stop_error_value = 0)

Y = com.sim(inputs)
[num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, targets)
plt.legend("Pima_Training Committee Gradient Descent block size 10\n [Cross validation] Total performance = " + str(total_performance) + "%")
plotroc(Y, targets, 1)


#Estimate on test set now
Y_test = com.sim(test_inputs)
for value in Y_test:
    print value[0]
    

plt.show()