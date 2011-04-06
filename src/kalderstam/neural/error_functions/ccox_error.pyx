# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
def derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots):
    """Eq. 14, derivative of Beta with respect to y(i)"""
    cdef double output, y_force, beta_out, res
    cdef int s, kronicker
    output = outputs[output_index]
    y_force = 0
    beta_out = np.exp(beta * output)
    for s in timeslots:
        #glogger.debugPlot('Partition function', part_func[s], style = 'b+')
        kronicker = 0
        if s == output_index:
            kronicker = 1
        y_force += kronicker - beta_out / part_func[s] * (1 + beta * (output - weighted_avg[s]))

    res = -y_force / beta_force
    #glogger.debugPlot('Beta derivative', res, style = 'r+')
    return res

@cython.boundscheck(False) # turn of bounds-checking for entire function    
def get_slope(double beta, risk_outputs, beta_risk, np.ndarray[np.float64_t, ndim=1] part_func, np.ndarray[np.float64_t, ndim=1] weighted_avg, np.ndarray[np.float64_t, ndim=2] outputs, np.ndarray[np.int_t, ndim=1] timeslots):
    cdef Py_ssize_t time_index, s
    cdef np.float64_t output, result = 0
    for time_index in range(timeslots.shape[0]):
        s = timeslots[time_index]
        output = outputs[s, 0]
        risk_outputs[s] = get_risk_outputs(time_index, timeslots, outputs)
        try:
            beta_risk[s] = np.exp(beta * risk_outputs[s])
        except FloatingPointError:
            #logger.error("In get_slope for calc_beta: \n if beta is 40 and risk_output is -23, we will get an underflow.\n Setting numpy.seterr(under = 'warn') or 'ignore', will do set it to zero in that case.")
            raise(FloatingPointError)

        part_func[s] = beta_risk[s].sum()
        weighted_avg[s] = (beta_risk[s] * risk_outputs[s]).sum() / part_func[s]
        result += (output - weighted_avg[s])

    return result
    
@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef np.ndarray[np.float64_t, ndim=1] get_risk_outputs(int time_index, timeslots, outputs):
    """s corresponds to the index of an output in outputs"""
    cdef int s, index, total_length = 0
    total_length = timeslots.shape[0]
    risk_outputs = np.zeros(total_length - time_index, dtype=np.float)
    for index in range(time_index, total_length):
        s = timeslots[index]
        risk_outputs[index - time_index] = outputs[s, 0]
    return risk_outputs