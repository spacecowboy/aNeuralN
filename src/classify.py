#!/usr/bin/env python

import matplotlib.pyplot as plt
from kalderstam.neural.training_functions import train_evolutionary,\
    traingd_block, train_committee
from kalderstam.util.filehandling import read_data_file, load_network,\
    parse_file, save_committee
from kalderstam.neural.network import build_feedforward_committee
from kalderstam.neural.matlab_functions import stat, plot2d2c, plotroc
import logging
from kalderstam.util.decorators import benchmark
    
def find_solution(P, T):
                
    #test, validation = get_validation_set(P, T, validation_size = 0.33)
    #net = build_feedforward(6, 3, 1)
    com = build_feedforward_committee(size = 10, input_number = len(P[0]), hidden_number = 20, output_number = len(T[0]))
    
    epochs = 7000
    
    print("Training...")
    #benchmark(train_committee)(com, train_evolutionary, P, T, 5, random_range = 3)
    benchmark(train_committee)(com, traingd_block, P, T, epochs, learning_rate = 0.03, block_size = 30)
    
    #P, T = test
    Y = com.sim(P)
    area = plotroc(Y, T, 1)
    
    #P, T = validation
    #Y = com.sim(P)
    #plotroc(Y, T, 2)
    
#    print("")
#    print("Stats for cut = 0.5")
#    [num_correct_first, num_correct_second, total_performance, num_first, num_second, missed] = stat(Y, T)
    
    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_gdblock20_rocarea" + str(area) + ".ann")
    #save_network(best, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_genetic_rocarea" + str(area) + ".ann")
    save_committee(com, "/export/home/jonask/Projects/aNeuralN/ANNs/classification_gdblock30_rocarea" + str(area) + ".anncom")
    
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
    P, T = parse_file("/home/gibson/jonask/Dropbox/Ann-Survival-Phd/Two_thirds_of_SA_1889_dataset.txt", targetcols = [5], ignorecols = [0,1,3,4])
    
    find_solution(P, T)
    
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_gdblock20_79.6664019063.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_gdblock20_78.9515488483.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Normalized breast/classification_genetic_80.2223987292.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Unnormalized breast/classification_genetic_80.4606830818.ann")
    #show_solution(P, T, "/export/home/jonask/Projects/aNeuralN/ANNs/Unnormalized breast/classification_gdblock20_79.1104050834.ann")
    
    