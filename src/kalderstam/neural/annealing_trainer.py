from random import random
from network import network
        
#Returns true if output is better
def evaluate_result(output):
    print 'result good or bad?'
    print 'return true false'
    
def modify_weights(node_list, temperature):
    for node in node_list:
        if random() < temperature:
            for key in node.weights.keys():
                node.weights[key] = random()
        
def save_weights(node_list):
    for node in node_list:
        for key in node.weights.keys():
            node.old_weights[key] = node.weights[key]
            
def restore_weights(node_list):
    for node in node_list:
        for key in node.weights.keys():
            node.weights[key] = node.old_weights[key]
            
net = network(2)
#random.seed()
initial_temperature = 1.0
temperature = initial_temperature
best_score = 0
while temperature > 0.01:
    node_list = net.get_all_nodes()
    modify_weights(node_list, temperature)
    result_list = []
    for i in range(10):
        result_list += net.update([i,i])
    score = sum(result_list)
    result_list2 = []
    for i in range(10):
        result_list2 += net.update([i, i+1])
    score += -sum(result_list2)
    if score > best_score:
        print 'Weights saved for score:', str(score)
        best_score = score
        save_weights(node_list)
    else:
        print 'Weights restored for score', str(score)
        restore_weights(node_list)
    
    temperature -= 0.01
    
print 'Best score: ', best_score
    
