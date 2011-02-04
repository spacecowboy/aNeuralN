import numpy
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