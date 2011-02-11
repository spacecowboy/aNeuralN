import numpy
from kalderstam.neural import network
import re
from kalderstam.neural.activation_functions import get_function
def read_data_file(filename):
    """Columns are data dimensions, rows are sample data. Whitespace separates the columns. Returns a python list [][]."""
    f = open(filename, 'r')
    inputs = []
    
    for line in f.readlines():
        cols = line.split() #Get rid of whitespace delimiters
        inputs.append(cols)
    
    return inputs

def parse_file(filename, targetcolumn, targetnum=1):
    return parse_data(numpy.array(read_data_file(filename)), targetcolumn, targetnum)
    

def parse_data(inputs, targetcolumn, targetnum=1):
    """inputs is an array of input and target columns. targetcolumn specifies which column is the starting target column (starting at 0!). targetnum is the number of target columns."""
    targets = numpy.array(inputs[:, targetcolumn:(targetcolumn + targetnum)], dtype='float64') #target is 40th column
    inputs = numpy.array(inputs[:, :targetcolumn], dtype='float64') #first 39 columns are inputs
    
    return (inputs, targets)

def save_network(net, filename = None):
    """If Filename is None, create a new as net_#hashnumber.ann"""
    if not filename:
        filename = "net_" + str(hash(net)) + ".ann"
    
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
    nodes = dict()
    node_weights = dict()
    
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
                    nodes[current_node] = network.node(active=function, bias = value)
                    node_weights[current_node] = dict()
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
    for node_name, node in nodes.iteritems():
        """Not for inputs"""
        if node_name.startswith("input"):
            net.num_of_inputs += 1
        else:
            for name, weight in node_weights[node_name].iteritems():
                node.connect_node(nodes[name], weight)
            """ add to network"""
            if node_name.startswith("hidden"):
                net.hidden_nodes.append(nodes[current_node])
            else:
                net.output_nodes.append(nodes[current_node])
            
    """Done! return net!"""
    return net
                    
    
if __name__ == '__main__':   
    from kalderstam.neural.network import build_feedforward
    net = build_feedforward()
     
    results = net.update([1, 2])
    
    print results
    
    filename = "/home/gibson/jonask/test.ann"
    print "saving and reloading"
    save_network(net, filename)
    
    net = load_network(filename)
    results = net.update([1, 2])
    print results