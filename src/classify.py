#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
from kalderstam.util.filehandling import read_data_file, load_network,\
    parse_file, save_network
from kalderstam.neural.network import network, build_feedforward
from kalderstam.neural.matlab_functions import stat, plot2d2c, plotroc
import logging
from kalderstam.neural.training_functions import traingd, train_evolutionary,\
    traingd_block
    
def find_solution(P, T):
                
    net = build_feedforward(6, 3, 1)
    
    epochs = 10
    
    #best = traingd(net, P, T, epochs)
    best = traingd_block(net, P, T, epochs, block_size=20)
    #best = train_evolutionary(net, P, T, epochs, random_range=5)
    
    Y = best.sim(P)
    plotroc(Y, T)
    
    print("")
    print("Stats for cut = 0.5")
    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    
    save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_gdblock20_" + str(total_performance) + ".ann")
    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_genetic_" + str(total_performance) + ".ann")
    
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
    
    P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt", 5, ignorecols = [0,1,4], ignorerows = 0)
    
    find_solution(P, T)
    
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_gdblock20_79.6664019063.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_gdblock20_78.9515488483.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_genetic_80.2223987292.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Unnormalized breast/classification_genetic_80.4606830818.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Unnormalized breast/classification_gdblock20_79.1104050834.ann")
    
    