# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
def derivative_beta(beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups):
    """Eq. 14, derivative of Beta with respect to y(i)"""
    cdef double output, y_force, beta_out, res
    cdef int s, kronicker
    output = outputs[output_index]
    y_force = 0
    beta_out = np.exp(beta * output)
    #for index in range(len(timeslots)):
    cdef int length = timeslots.shape[0]
    for index from 0 <= index < length:
        s = timeslots[index]
        kronicker = 0
        if s == output_index:
            kronicker = 1
        if output_index in risk_groups[index]: #If output_index is not in the risk group, dy_part is zero (and kronicker-delta must also be zero of course), so no need to waste computation
            y_force += kronicker - beta_out / part_func[index] * (1 + beta * (output - weighted_avg[index]))
        else:
            y_force += kronicker

    res = -y_force / beta_force
    #glogger.debugPlot('Beta derivative', res, style = 'r+')
    return res

@cython.boundscheck(False) # turn of bounds-checking for entire function    
def get_slope(double beta, risk_groups, beta_risk, np.ndarray[np.float64_t, ndim=1] part_func, np.ndarray[np.float64_t, ndim=1] weighted_avg, np.ndarray[np.float64_t, ndim=2] outputs, np.ndarray[np.int_t, ndim=1] timeslots):
    cdef Py_ssize_t time_index, s
    cdef np.float64_t output, result = 0
    cdef int length = timeslots.shape[0]
    for time_index from 0 <= time_index < length:
        s = timeslots[time_index]
        output = outputs[s, 0]
        risk_outputs = outputs[risk_groups[time_index], 0]
        try:
            beta_risk[time_index] = np.exp(beta * risk_outputs)
        except FloatingPointError:
            #logger.error("In get_slope for calc_beta: \n if beta is 40 and risk_output is -23, we will get an underflow.\n Setting numpy.seterr(under = 'warn') or 'ignore', will do set it to zero in that case.")
            raise(FloatingPointError)

        part_func[time_index] = beta_risk[time_index].sum()
        weighted_avg[time_index] = (beta_risk[time_index] * risk_outputs).sum() / part_func[time_index]
        result += (output - weighted_avg[time_index])

    return result