import numpy
from kalderstam.neural import network
import re
from kalderstam.neural.activation_functions import get_function
from os import path

def read_data_file(filename):
    """Columns are data dimensions, rows are sample data. Whitespace separates the columns. Returns a python list [[]]."""
    with open(filename, 'r') as f:
        inputs = [line.split() for line in f.readlines()]
    
    return inputs

def parse_file(filename, targetcols, inputcols = None, ignorecols = [], ignorerows = []):
    return parse_data(numpy.array(read_data_file(filename)), targetcols, inputcols, ignorecols, ignorerows)

def parse_data(inputs, targetcols, inputcols = None, ignorecols = [], ignorerows = [], normalize = True):
    """inputs is an array of data columns. targetcols is either an int describing which column is a the targets or it's a list of several ints pointing to multiple target columns.
    Input columns follows the same pattern, but are not necessary if the inputs are all that's left when target columns are subtracted.
    Ignorecols can be used instead if it's easier to specify which columns to ignore instead of which are inputs.
    Ignorerows specify which, if any, rows should be skipped."""
    
    inputs = numpy.delete(inputs, ignorerows, 0)
    
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
                
    
    return (inputs, targets)

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
            """Write its activation function"""
            f.write("function=" + str(node.function) + "\n")
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
                
                """check function name"""
                m = re.search('function\s*=\s*([\w\d]+)', line)
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
    from kalderstam.neural.network import build_feedforward
    net = build_feedforward()
     
    results = net.update([1, 2])
    
    print results
    
    filename = path.join(path.expanduser("~"), "test.ann")
    print "saving and reloading"
    save_network(net, filename)
    
    net = load_network(filename)
    results = net.update([1, 2])
    print results
