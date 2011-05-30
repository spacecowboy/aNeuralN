from os import path
from kalderstam.util.filehandling import parse_file, save_committee
from kalderstam.neural.network import build_feedforward_committee, \
    build_feedforward
from kalderstam.util.decorators import benchmark
import logging
from kalderstam.matlab.matlab_functions import plotroc, stat, \
    get_rocarea_and_best_cut
import matplotlib.pyplot as plt
from kalderstam.neural.training.committee import train_committee
from kalderstam.neural.training.genetic import train_evolutionary
from kalderstam.neural.training.gradientdescent import traingd

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

#net = build_feedforward(8, 8, 1)

epochs = 10

#best = benchmark(train_evolutionary)(net, test, validation, 10, random_range = 1)
#best = benchmark(traingd_block)(net, test, validation, epochs, block_size = 10, stop_error_value = 0)

com = build_feedforward_committee(size = 10, input_number = 8, hidden_number = 8, output_number = 1)

print "Training evolutionary..."
benchmark(train_committee)(com, train_evolutionary, inputs, targets, epochs, random_range = 1)

Y = com.sim(inputs)
area, best_cut = get_rocarea_and_best_cut(Y, targets)
[num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, targets, cut = best_cut)
print("Total number of data: " + str(len(targets)) + " (" + str(num_second) + " ones and " + str(num_first) + " zeros)")
print("Number of misses: " + str(missed) + " (" + str(total_performance) + "% performance)")
print("Specificity: " + str(num_correct_first) + "% (Success for class 0)")
print("Sensitivity: " + str(num_correct_second) + "% (Success for class 1)")
print("Roc Area: " + str(area) + "%")

save_committee(com, "/export/home/jonask/Projects/aNeuralN/ANNs/pimatrain_gen__rocarea" + str(area) + ".anncom")

print "\nTraining with gradient descent..."
benchmark(train_committee)(com, traingd, inputs, targets, epochs, block_size = 10, stop_error_value = 0)

Y = com.sim(inputs)
area, best_cut = get_rocarea_and_best_cut(Y, targets)
[num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, targets, cut = best_cut)
print("Total number of data: " + str(len(targets)) + " (" + str(num_second) + " ones and " + str(num_first) + " zeros)")
print("Number of misses: " + str(missed) + " (" + str(total_performance) + "% performance)")
print("Specificity: " + str(num_correct_first) + "% (Success for class 0)")
print("Sensitivity: " + str(num_correct_second) + "% (Success for class 1)")
print("Roc Area: " + str(area) + "%")

Y = com.sim(inputs)
[num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, targets)
#plt.legend("Pima_Training Committee Gradient Descent block size 10\n [Cross validation] Total performance = " + str(total_performance) + "%")
area = plotroc(Y, targets, 1)


save_committee(com, "/export/home/jonask/Projects/aNeuralN/ANNs/pimatrain_rocarea" + str(area) + ".anncom")

#Estimate on test set now
print("\nPredictions for test set:")
Y_test = com.sim(test_inputs)
for value in Y_test:
    print value[0]

plt.show()
