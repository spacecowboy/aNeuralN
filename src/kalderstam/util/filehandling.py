import numpy
from kalderstam.neural import network
import re
from kalderstam.neural.activation_functions import get_function
from os import path
from random import random

def read_data_file(filename):
    """Columns are data dimensions, rows are sample data. Whitespace separates the columns. Returns a python list [[]]."""
    with open(filename, 'r') as f:
        inputs = [line.split() for line in f.readlines()]
    
    return inputs

def parse_file(filename, targetcols = None, inputcols = None, ignorecols = [], ignorerows = [], normalize = True):
    return parse_data(numpy.array(read_data_file(filename)), targetcols, inputcols, ignorecols, ignorerows, normalize)

def parse_data(inputs, targetcols = None, inputcols = None, ignorecols = [], ignorerows = [], normalize = True):
    """inputs is an array of data columns. targetcols is either an int describing which column is a the targets or it's a list of several ints pointing to multiple target columns.
    Input columns follows the same pattern, but are not necessary if the inputs are all that's left when target columns are subtracted.
    Ignorecols can be used instead if it's easier to specify which columns to ignore instead of which are inputs.
    Ignorerows specify which, if any, rows should be skipped."""
    if targetcols == None:
        targetcols = []
    try:
        targetcols = [int(targetcols)]
    except TypeError:
        #targetcols is alreayd a list
        pass
    
    if not inputcols:
        inputcols = range(len(inputs[0]))
        destroycols = []
        try:
            destroycols.append(int(targetcols)) #Only if it's an int
        except TypeError:
            destroycols.extend(targetcols)
        try:
            destroycols.append(int(ignorecols)) #Only if it's an int
        except TypeError:
            destroycols.extend(ignorecols)
            
        inputcols = numpy.delete(inputcols, destroycols, 0)
        
    for line in range(len(inputs)):
        if len(targetcols) == 0:
            all_cols = inputs[line, inputcols]
        elif len(inputcols) == 0:
            all_cols = inputs[line, targetcols]
        else:
            all_cols = inputs[line, numpy.append(inputcols, targetcols)]
        for col in all_cols: #check only valid columns
            try:
                float(col)
            except ValueError: #This row contains crap, get rid of it
                ignorerows.append(line)
                break #skip to next line
    
    inputs = numpy.delete(inputs, ignorerows, 0)
    
    targets = numpy.array(inputs[:, targetcols], dtype = 'float64')
    inputs = numpy.array(inputs[:, inputcols], dtype = 'float64')
    
    if normalize:
        #First we must determine which columns have real values in them
        #Basically, we if it isn't a binary value by comparing to 0 and 1
        for col in range(len(inputs[0])):
            real = False
            for value in inputs[:, col]:
                if value != 0 and value != 1:
                    real = True
                    break #No point in continuing now that we know they're real
            if real:
                #Subtract the mean and divide by the standard deviation
                inputs[:, col] = (inputs[:, col] - numpy.mean(inputs[:, col])) / numpy.std(inputs[:, col])
    
    #Now divide the input into test and validation parts
    
    return inputs, targets

def get_validation_set(inputs, targets, validation_size = 0.2):
    if validation_size < 0 or validation_size > 1:
        raise TypeError('validation_size not between 0 and 1')
    test_inputs = []
    test_targets = []
    validation_inputs = []
    validation_targets = []
    for row in range(len(inputs)):
        if random() > validation_size:
            test_inputs.append(inputs[row])
            test_targets.append(targets[row])
        else:
            validation_inputs.append(inputs[row])
            validation_targets.append(targets[row])
    
    test_inputs = numpy.array(test_inputs, dtype = 'float64')
    test_targets = numpy.array(test_targets, dtype = 'float64')
    validation_inputs = numpy.array(validation_inputs, dtype = 'float64')
    validation_targets = numpy.array(validation_targets, dtype = 'float64')
    
    return ((test_inputs, test_targets), (validation_inputs, validation_targets))

def get_stratified_validation_set(inputs, targets, validation_size = 0.2):
    """Will sort on 0-1 for targets. Only valid for classification sets."""
    zero_inputs, zero_targets = [], []
    one_inputs, one_targets = [], []
    for input, target in zip(inputs, targets):
        target = numpy.append(numpy.array([]), target) #Support both [0, 1] and [[0], [1]]
        if target[0] < 1: #Array of output values, in this case only one output node exists
            zero_inputs.append(input)
            zero_targets.append(target)
        else:
            one_inputs.append(input)
            one_targets.append(target)
    #Values are now divided up. Now, choose a validation set by dividing up each set randomly
    (zero_test_inputs, zero_test_targets), (zero_validation_inputs, zero_validation_targets) = get_validation_set(zero_inputs, zero_targets, validation_size)
    (one_test_inputs, one_test_targets), (one_validation_inputs, one_validation_targets) = get_validation_set(one_inputs, one_targets, validation_size)
    #Numpy is crap at appending data
    test_inputs = list(zero_test_inputs)
    test_targets = list(zero_test_targets)
    validation_inputs = list(zero_validation_inputs)
    validation_targets = list(zero_validation_targets)
    #now we add the two sets together
    test_inputs.extend(one_test_inputs)
    test_targets.extend(one_test_targets)
    validation_inputs.extend(one_validation_inputs)
    validation_targets.extend(one_validation_targets)
    #Finally, convert back to numpy arrays and return
    test_inputs = numpy.array(test_inputs, dtype = 'float64')
    test_targets = numpy.array(test_targets, dtype = 'float64')
    validation_inputs = numpy.array(validation_inputs, dtype = 'float64')
    validation_targets = numpy.array(validation_targets, dtype = 'float64')
    
    return ((test_inputs, test_targets), (validation_inputs, validation_targets))

def save_committee(com, filename = None):
    """If Filename is None, create a new as net_#hashnumber.ann and save in home dir"""
    if not filename:
        filename = "com_" + str(hash(com)) + ".anncom"
        filename = path.join(path.expanduser("~"), filename)
    
    """Open a file to write to"""
    with open(filename, 'w') as f:
        net_number = 0
        for net in com.nets:
            f.write("<net_" + str(net_number) + ">\n")
            net_number += 1
            for node in net.get_all_nodes():
                node_type = "output_"
                node_index = 0
                if node in net.hidden_nodes:
                    node_type = "hidden_"
                    node_index = net.hidden_nodes.index(node)
                else:
                    node_index = net.output_nodes.index(node)
                """Write node identifier"""
                f.write("[" + node_type + str(node_index) + "]\n")
                """Write its activation activation_function"""
                f.write("activation_function=" + str(node.activation_function) + "\n")
                """Write its bias"""
                f.write("bias=" + str(node.bias) + "\n")
                """Now write its connections and weights"""
                for back_node, back_weight in node.weights.iteritems():
                    """Assume its a hidden node, but check if its actually an output node"""
                    type = "hidden_"
                    type_index = 0
                    try:
                        if back_node in net.output_nodes:
                            type = "output_"
                            type_index = net.output_nodes.index(back_node)
                        else:
                            type_index = net.hidden_nodes.index(back_node)
                    except ValueError:
                        """back_node is input actually"""
                        type = "input_"
                        type_index = back_node
                    f.write(type + str(type_index) + ":" + str(back_weight) + "\n")
                """End with empty line since it's nicer that way"""
                f.write("\n")
    """And we're done!"""
    
def load_committee(filename):
    """Create the committee"""
    com = network.committee()
    
    nodes = {}
    node_weights = {}
    
    """Current node we're working on"""
    current_net = None
    current_node = None
    function = None
    
    """Read file"""
    with open(filename, 'r') as f:
        """Parse row by row"""
        for line in f.readlines():
            """Skip empty lines"""
            if not re.search('^\s*$', line):
                
                """check if id for current_node"""
                m = re.search('\<(net_\d+)\>', line)
                if m:
                    """Create a network"""
                    current_net = network.network()
                    com.nets.append(current_net)
                    nodes[current_net] = {}
                    node_weights[current_net] = {}
                    continue
                
                """check if id for current_node"""
                m = re.search('\[(\w+_\d+)\]', line)
                if m:
                    current_node = m.group(1)
                    continue
                
                """check activation_function name"""
                m = re.search('activation_function\s*=\s*([\w\d]+)', line)
                if m:
                    function = get_function(m.group(1))
                    continue
                
                """check bias"""
                m = re.search('bias\s*=\s*([-\d\.]*)', line)
                if m:
                    try:
                        value = float(m.group(1))
                    except:
                        value = None #Random value
                    """ create node"""
                    nodes[current_net][current_node] = network.node(active = function, bias = value)
                    node_weights[current_net][current_node] = {}
                    continue
                
                """check weights"""
                m = re.search('(\w+_\d+):([-\d\.]*)', line)
                if m:
                    back_node = m.group(1)
                    try:
                        weight = float(m.group(2))
                    except:
                        weight = None #Will yield random weight
                    node_weights[current_net][current_node][back_node] = weight
                    """If back_node is an input node, we have to create it down here if it doesn't exist"""
                    if back_node.startswith('input') and back_node not in nodes:
                        nodes[current_net][back_node] = int(back_node.strip('input_'))
                    continue
    
    """Now iterate over the hashes and connect the nodes for real"""
    for net in com.nets:
        for node_name, node in nodes[net].items():
            """Not for inputs"""
            if node_name.startswith("input"):
                net.num_of_inputs += 1
            else:
                for name, weight in node_weights[net][node_name].items():
                    node.connect_node(nodes[net][name], weight)
                """ add to network"""
                if node_name.startswith("hidden"):
                    net.hidden_nodes.append(nodes[net][node_name])
                else:
                    net.output_nodes.append(nodes[net][node_name])
            
    """Done! return committee!"""
    return com

def save_network(net, filename = None):
    """If Filename is None, create a new as net_#hashnumber.ann and save in home dir"""
    if not filename:
        filename = "net_" + str(hash(net)) + ".ann"
        filename = path.join(path.expanduser("~"), filename)
    
    """Open a file to write to"""
    with open(filename, 'w') as f:
        for node in net.get_all_nodes():
            node_type = "output_"
            node_index = 0
            if node in net.hidden_nodes:
                node_type = "hidden_"
                node_index = net.hidden_nodes.index(node)
            else:
                node_index = net.output_nodes.index(node)
            """Write node identifier"""
            f.write("[" + node_type + str(node_index) + "]\n")
            """Write its activation activation_function"""
            f.write("activation_function=" + str(node.activation_function) + "\n")
            """Write its bias"""
            f.write("bias=" + str(node.bias) + "\n")
            """Now write its connections and weights"""
            for back_node, back_weight in node.weights.iteritems():
                """Assume its a hidden node, but check if its actually an output node"""
                type = "hidden_"
                type_index = 0
                try:
                    if back_node in net.output_nodes:
                        type = "output_"
                        type_index = net.output_nodes.index(back_node)
                    else:
                        type_index = net.hidden_nodes.index(back_node)
                except ValueError:
                    """back_node is input actually"""
                    type = "input_"
                    type_index = back_node
                f.write(type + str(type_index) + ":" + str(back_weight) + "\n")
            """End with empty line since it's nicer that way"""
            f.write("\n")
    """And we're done!"""
    
def load_network(filename):
    """Create a network"""
    net = network.network()
    nodes = {}
    node_weights = {}
    
    """Read file"""
    with open(filename, 'r') as f:
        """Current node we're working on"""
        current_node = None
        function = None
        """Parse row by row"""
        for line in f.readlines():
            """Skip empty lines"""
            if not re.search('^\s*$', line):
                
                """check if id for current_node"""
                m = re.search('\[(\w+_\d+)\]', line)
                if m:
                    current_node = m.group(1)
                    continue
                
                """check activation_function name"""
                m = re.search('activation_function\s*=\s*([\w\d]+)', line)
                if m:
                    function = get_function(m.group(1))
                    continue
                
                """check bias"""
                m = re.search('bias\s*=\s*([-\d\.]*)', line)
                if m:
                    try:
                        value = float(m.group(1))
                    except:
                        value = None #Random value
                    """ create node"""
                    nodes[current_node] = network.node(active = function, bias = value)
                    node_weights[current_node] = {}
                    continue
                
                """check weights"""
                m = re.search('(\w+_\d+):([-\d\.]*)', line)
                if m:
                    back_node = m.group(1)
                    try:
                        weight = float(m.group(2))
                    except:
                        weight = None #Will yield random weight
                    node_weights[current_node][back_node] = weight
                    """If back_node is an input node, we have to create it down here if it doesn't exist"""
                    if back_node.startswith('input') and back_node not in nodes:
                        nodes[back_node] = int(back_node.strip('input_'))
                    continue
    
    """Now iterate over the hashes and connect the nodes for real"""
    for node_name, node in nodes.items():
        """Not for inputs"""
        if node_name.startswith("input"):
            net.num_of_inputs += 1
        else:
            for name, weight in node_weights[node_name].items():
                node.connect_node(nodes[name], weight)
            """ add to network"""
            if node_name.startswith("hidden"):
                net.hidden_nodes.append(nodes[node_name])
            else:
                net.output_nodes.append(nodes[node_name])
            
    """Done! return net!"""
    return net
                    
    
if __name__ == '__main__':   
    print("Testing network saving/loading")
    
    from kalderstam.neural.network import build_feedforward, build_feedforward_committee
    net = build_feedforward()
     
    results1 = net.update([1, 2])
    
    print results1
    
    filename = path.join(path.expanduser("~"), "test.ann")
    print "saving and reloading"
    save_network(net, filename)
    
    net = load_network(filename)
    results2 = net.update([1, 2])
    print results2
    
    assert(abs(results1[0] - results2[0]) < 0.0001) #float doesn't handle absolutes so well
    print("Good, now testing committee...")
    
    com = build_feedforward_committee()
    results1 = com.update([1, 2])
    print results1
    
    filename = path.join(path.expanduser("~"), "test.anncom")
    print "saving and reloading"
    
    save_committee(com, filename)
    
    com = load_committee(filename)
    results2 = com.update([1, 2])
    print results2
    
    assert(abs(results1[0] - results2[0]) < 0.0001) #float doesn't handle absolutes so well
    
    print("Results are good. Testing input parsing....")
    filename = path.join(path.expanduser("~"), "ann_input_data_test_file.txt")
    print("First, split the file into a test set(80%) and validation set(20%)...")
    inputs, targets = parse_file(filename, targetcols = 5, ignorecols = [0,1,4], ignorerows = [])
    test, validation = get_validation_set(inputs, targets, validation_size=0.5)
    print(len(test[0]))
    print(len(test[1]))
    print(len(validation[0]))
    print(len(validation[1]))
    assert(len(test) == 2)
    assert(len(test[0]) > 0)
    assert(len(test[1]) > 0)
    assert(len(validation) == 2)
    assert(len(validation[0]) > 0)
    assert(len(validation[1]) > 0)
    print("Went well, now expecting a zero size validation set...")
    test, validation = get_validation_set(inputs, targets, validation_size=0)
    print(len(test[0]))
    print(len(test[1]))
    print(len(validation[0]))
    print(len(validation[1]))
    assert(len(test) == 2)
    assert(len(test[0]) > 0)
    assert(len(test[1]) > 0)
    assert(len(validation) == 2)
    assert(len(validation[0]) == 0)
    assert(len(validation[1]) == 0)
    print("As expected. Now a 100% validation set...")
    test, validation = get_validation_set(inputs, targets, validation_size=1)
    print(len(test[0]))
    print(len(test[1]))
    print(len(validation[0]))
    print(len(validation[1]))
    assert(len(test) == 2)
    assert(len(test[0]) == 0)
    assert(len(test[1]) == 0)
    assert(len(validation) == 2)
    assert(len(validation[0]) > 0)
    assert(len(validation[1]) > 0)
    print("Now we test a stratified set...")
    test, validation = get_stratified_validation_set(inputs, targets, validation_size = 0.5)
    print(len(test[0]))
    print(len(test[1]))
    print(len(validation[0]))
    print(len(validation[1]))
    assert(len(test) == 2)
    assert(len(test[0]) > 0)
    assert(len(test[1]) > 0)
    assert(len(validation) == 2)
    assert(len(validation[0]) > 0)
    assert(len(validation[1]) > 0)
    print("Test with no targets, the no inputs")
    inputs, targets = parse_file(filename, ignorecols = [0,1,4], ignorerows = [])
    assert((targets.size) == 0)
    assert((inputs.size) > 0)
    inputs, targets = parse_file(filename, targetcols = 3, ignorecols = [0,1, 2,4, 5, 6,7 ,8, 9], ignorerows = [])
    assert((targets.size) > 0)
    assert((inputs.size) == 0)
    
    print("All tests completed successfully!")
    
