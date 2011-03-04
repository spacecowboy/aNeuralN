import sys
from random import sample
import numpy

def split_file(filename, target_column = None, ignorecols = None):
    with open(filename, 'r') as f:
        data = [line.split() for line in f.readlines()]
    if target_column:
        training, validation = split_proportionally(data, target_column, ignorecols)
    else:
        training, validation = split_randomly(data)
    return (training, validation)
    
def split_proportionally(data, target_column, ignorecols = None, validation_size = 0.25):
    data = numpy.array(data)
    
    if ignorecols != None:
        inputcols = range(len(data[0]))
        destroycols = []
        try:
            destroycols.append(int(ignorecols)) #Only if it's an int
        except TypeError:
            destroycols.extend(ignorecols)
            
        inputcols = numpy.delete(inputcols, destroycols, 0)
    
        data = numpy.array(data[:, inputcols], dtype = 'float64')
    
        target_column -= len(destroycols)
    #Find the average and split into two sets
    avg = data.mean(axis=0)[target_column]
    
    #Boolean vectors
    left = data[:, target_column] < avg
    right = data[:, target_column] >= avg
    
    left = data[left, :]
    right = data[right, :]
    
    left_proportion = float(len(left))/(len(left) + len(right))
    right_proportion = float(len(right))/(len(left) + len(right))
    
    validation = sample(left.tolist(), int(round(left_proportion*validation_size*len(data))))
    validation += sample(right.tolist(), int(right_proportion*validation_size*len(data)))
    
    training = numpy.array([line for line in data.tolist() if line not in validation])
    validation = numpy.array(validation)
    
    return (training, validation)

def split_randomly(data, validation_size = 0.25):
    #Matlab's train function divides it to 60% train set, 20% test set, 20% validation set.
    validation = numpy.array(sample(data, int(round(validation_size*len(data)))))
    training = numpy.array([line for line in data if line not in validation])
    return (training, validation)

if __name__ == '__main__':
    testlist = range(100)
    t, v = split_randomly(testlist)
    print("normal list, total length: " + str(len(testlist)) + ", t length: " + str(len(t)) + ", v length: " + str(len(v)))
    
    testlist = numpy.array(range(100))
    t, v = split_randomly(testlist)
    print("\nnumpy list, total length: " + str(len(testlist)) + ", t length: " + str(len(t)) + ", v length: " + str(len(v)))
    print("numpy list, total mean: " + str(testlist.mean(axis=0)) + ", t mean: " + str(t.mean(axis=0)) + ", v mean: " + str(v.mean(axis=0)))
    
    
    testlist = [[i, sample([0,1], 1)[0]] for i in range(100)]
    t, v = split_proportionally(testlist, 1)
    print("\nnumpy list, total length: " + str(len(testlist)) + ", t length: " + str(len(t)) + ", v length: " + str(len(v)))
    print("numpy list, total mean: " + str(numpy.array(testlist).mean(axis=0)) + ", t mean: " + str(t.mean(axis=0)) + ", v mean: " + str(v.mean(axis=0)))
    
    testlist = numpy.array([[i, sample([0,1], 1)[0]] for i in range(100)])
    t, v = split_proportionally(testlist, 1)
    print("\nnumpy list, total length: " + str(len(testlist)) + ", t length: " + str(len(t)) + ", v length: " + str(len(v)))
    print("numpy list, total mean: " + str(testlist.mean(axis=0)) + ", t mean: " + str(t.mean(axis=0)) + ", v mean: " + str(v.mean(axis=0)))
    
    
    if len(sys.argv) < 2:
        print('No file specified...')
    else:
        if len(sys.argv) < 3:
            t, v = split_file(sys.argv[1])
        else:
            t, v = split_file(sys.argv[1], int(sys.argv[2]), 0)
        print("\nTraining set")
        for line in t:
            s = ""
            for col in line:
                s += (str(col) + "\t")
            print(s)
        print("\nValidation set")
        for line in v:
            s = ""
            for col in line:
                s += (str(col) + "\t")
            print(s)