#!/usr/bin/env python

import matplotlib.pyplot as plt
from kalderstam.util.filehandling import read_data_file, load_network,\
    parse_file, save_committee
from kalderstam.neural.network import build_feedforward_committee, build_feedforward
from kalderstam.matlab.matlab_functions import stat, plot2d2c, plotroc, loadsyn1, loadsyn2, loadsyn3
from kalderstam.neural.training.gradientdescent import traingd
from kalderstam.neural.training.genetic import train_evolutionary
from kalderstam.neural.training.committee import train_committee
import logging
from kalderstam.util.decorators import benchmark
from kalderstam.util.filehandling import get_stratified_validation_set,\
    get_validation_set
    
def find_solution(P, T):
                
    #test, validation = get_validation_set(P, T, validation_size = 0.33)
    net = build_feedforward(input_number = len(P[0]), hidden_number = 4, output_number = len(T[0]))
    #com = build_feedforward_committee(size = 4, input_number = len(P[0]), hidden_number = 6, output_number = len(T[0]))
    
    epochs = 1000
    
    testset, valset = get_validation_set(P, T, validation_size = 0.01)
    
    print("Training...")
    net = benchmark(train_evolutionary)(net, testset, valset, 100, random_range = 1)
    net = benchmark(traingd)(net, testset, valset, epochs, learning_rate = 0.1, block_size = 1)
    
    #benchmark(train_committee)(com, train_evolutionary, P, T, 100, random_range = 1)
    #benchmark(train_committee)(com, traingd, P, T, epochs, learning_rate = 0.1, block_size = 30)
    
    #P, T = test
    Y = net.sim(P)
    area, best_cut = plotroc(Y, T, 1)
    plot2d2c(net, P, T, figure = 2, cut = best_cut)
    
    #P, T = validation
    #Y = com.sim(P)
    #plotroc(Y, T, 2)
    
#    print("")
#    print("Stats for cut = 0.5")
#    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    
    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_gdblock20_rocarea" + str(area) + ".ann")
    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_genetic_rocarea" + str(area) + ".ann")
    #save_committee(com, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_gdblock30_rocarea" + str(area) + ".anncom")
    #save_committee(com, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_genetic_rocarea" + str(area) + ".anncom")
    
    plt.show()

def show_solution(P, T, path):
    best = load_network(path)
    Y = best.sim(P)
    area = plotroc(Y, T)
    print("")
    print("Stats for cut = 0.5")
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    plt.title(path + "\nArea = " + str(area))
    plt.show()
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('classify')
    
    #P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt", targetcols = [5], ignorecols = [0,1,4])
    #P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt", targetcols = [5], ignorecols = [0,1,3,4])
    P, T = loadsyn3(100)
    
    find_solution(P, T)
    
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_gdblock20_79.6664019063.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_gdblock20_78.9515488483.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_genetic_80.2223987292.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Unnormalized breast/classification_genetic_80.4606830818.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Unnormalized breast/classification_gdblock20_79.1104050834.ann")
    
    