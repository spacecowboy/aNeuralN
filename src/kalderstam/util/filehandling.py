def read_data_file(filename):
    """Columns are data dimensions, rows are sample data. Whitespace separates the columns. Returns a python list [][]."""
    f = open(filename, 'r')
    inputs = []
    
    for line in f.readlines():
        cols = line.split() #Get rid of whitespace delimiters
        inputs.append(cols)
    
    return inputs