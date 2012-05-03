'''
Some utility methods that SHOULD exist in numpy...
'''

def indexOf(array, item):
    j = None
    for i in xrange(len(array)):
        #Search inside nested arrays
        if len(array.shape) > 1:
            j = indexOf(array[i], item)
            if j is not None:
                indices = [i]
                try:
                    indices.extend(j)
                except TypeError:
                    indices.append(j)
                return tuple(indices)
        else:
            #Compare values
            if array[i] == item:
                return i
    return None
