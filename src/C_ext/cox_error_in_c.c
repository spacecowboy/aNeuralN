#include <Python.h> // Must be the first line!
#include <numpy/arrayobject.h> // NumPy as seen from C
#include <math.h>

int inRiskGroup(long output_index, PyArrayObject *risk_group) {
	if (PyArray_NDIM(risk_group) != 1 || PyArray_TYPE(risk_group) != NPY_INT64) {
		PyErr_Format(PyExc_ValueError, "Because risk_group array is %d-dimensional or not of type int64", PyArray_NDIM(risk_group));
		return -1;
	}

	long index, max, risk_index;
	int member = 0;

	max = PyArray_DIM(risk_group, 0);

	for (index = 0; index < max; index++) {
		risk_index = *(long *) PyArray_GETPTR1(risk_group, index);
		if (risk_index == output_index) {
			member = 1;
			break;
		}
	}

	return member;
}

static PyObject *derivative_beta(PyObject *self, PyObject *args)
{
	// Define the arguments, a couple of arrays and a few doubles and integers.
	PyArrayObject *timeslots, *outputs, *part_func, *weighted_avg, *risk_group; // These are all indexed by slot_index.
	//PyListObject *risk_groups; // python list
	PyObject *pybeta, *pybeta_force, *pyoutput_index, *risk_groups;
	
	long es, output_index, slot_index, slotmax;
	double z, w, beta, beta_force, output, beta_out, y_force, result, kronicker;

	// Order of the arguments are: beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups
	if (!PyArg_ParseTuple(args, "OOOOOOOO", &pybeta, &part_func, &weighted_avg, &pybeta_force, &pyoutput_index, &outputs, &timeslots, &risk_groups))
	{ 
		PyErr_Format(PyExc_ValueError, "Order of the arguments are: beta, part_func, weighted_avg, beta_force, output_index, outputs, timeslots, risk_groups");
		return NULL;
	}

	// Check length and type of the array, return exception if wrong.
	if (PyArray_NDIM(timeslots) != 1 || PyArray_TYPE(timeslots) != NPY_INT64) {
		PyErr_Format(PyExc_ValueError, "Because timeslots array is %d-dimensional or not of type int64", PyArray_NDIM(timeslots));
		return NULL;
	} else if (PyArray_NDIM(outputs) != 2 || PyArray_TYPE(outputs) != NPY_DOUBLE) {
		PyErr_Format(PyExc_ValueError, "Because outputs array is %d-dimensional or not of type double", PyArray_NDIM(outputs));
		return NULL;
	} else if (PyList_Size((PyObject*) risk_groups) != PyArray_DIM(timeslots,0)) {
		PyErr_Format(PyExc_ValueError, "Because risk_groups array is not the same length as the timeslots array.");
		return NULL;
	} else if (PyArray_NDIM(part_func) != 1 || PyArray_TYPE(part_func) != NPY_DOUBLE) {
		PyErr_Format(PyExc_ValueError, "Because part_func array is %d-dimensional or not of type double", PyArray_NDIM(part_func));
		return NULL;
	} else if (PyArray_NDIM(weighted_avg) != 1 || PyArray_TYPE(weighted_avg) != NPY_DOUBLE) {
		PyErr_Format(PyExc_ValueError, "Because weighted_avg array is %d-dimensional or not of type double", PyArray_NDIM(weighted_avg));
		return NULL;
	}
	// Now convert the non-lists to C-types
	beta = PyFloat_AS_DOUBLE(pybeta);
	beta_force = PyFloat_AS_DOUBLE(pybeta_force);
	output_index = PyLong_AsLong(pyoutput_index);

	// Actual algorithm follows
	output = *(double *) PyArray_GETPTR2(outputs, output_index, 0);
	beta_out = exp(beta * output);
	y_force = 0;
	//for es, risk_group, z, w in zip(timeslots, risk_groups, part_func, weighted_avg):
	slotmax = PyArray_DIM(timeslots, 0);
	for (slot_index = 0; slot_index < slotmax; slot_index++) {
		es = *(long *) PyArray_GETPTR1(timeslots, slot_index);
		z = *(double *) PyArray_GETPTR1(part_func, slot_index);
		w = *(double *) PyArray_GETPTR1(weighted_avg, slot_index);
		risk_group = (PyArrayObject*) PyList_GetItem(risk_groups, slot_index);
		
		kronicker = 0;
		if (es == output_index) {
			kronicker = 1;
		}
		if (inRiskGroup(output_index, risk_group)) {
			y_force += kronicker - beta_out / z * (1 + beta * (output - w));
		} else {
			y_force += kronicker;
		}
	}

	result = -y_force / beta_force;
	
	return Py_BuildValue("d", result); // Build the python type out of the C-result
};


static PyObject *get_slope(PyObject *self, PyObject *args)
{
	// Define the arguments, a couple of arrays and a few doubles and integers.
	PyArrayObject *timeslots, *outputs, *part_func, *weighted_avg, *risk_group, *beta_risk_outputs; // These are all indexed by slot_index. Potential sub-arrays are indexed by risk_index
	//PyListObject *risk_groups; // python list
	PyObject *pybeta, *risk_groups, *beta_risks;
	
	long es, slotmax, slot_index, risk_index, riskmax, risk_output_index;
	double *z, *w, beta, output, *beta_risk, result, risk_output;

	// Order of the arguments are: beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots
	if (!PyArg_ParseTuple(args, "OOOOOOO", &pybeta, &risk_groups, &beta_risks, &part_func, &weighted_avg, &outputs, &timeslots))
	{ 
		PyErr_Format(PyExc_ValueError, "Order of the arguments are: beta, risk_groups, beta_risk, part_func, weighted_avg, outputs, timeslots");
		return NULL;
	}

	// Check length and type of the array, return exception if wrong.
	if (PyArray_NDIM(timeslots) != 1 || PyArray_TYPE(timeslots) != NPY_INT64) {
		PyErr_Format(PyExc_ValueError, "Because timeslots array is %d-dimensional or not of type int64", PyArray_NDIM(timeslots));
		return NULL;
	} else if (PyArray_NDIM(outputs) != 2 || PyArray_TYPE(outputs) != NPY_DOUBLE) {
		PyErr_Format(PyExc_ValueError, "Because outputs array is %d-dimensional or not of type double", PyArray_NDIM(outputs));
		return NULL;
	} else if (PyList_Size(risk_groups) != PyArray_DIM(timeslots,0)) {
		PyErr_Format(PyExc_ValueError, "Because risk_groups array is not the same length as the timeslots array.");
		return NULL;
	} else if (PyArray_NDIM(part_func) != 1 || PyArray_TYPE(part_func) != NPY_DOUBLE) {
		PyErr_Format(PyExc_ValueError, "Because part_func array is %d-dimensional or not of type double", PyArray_NDIM(part_func));
		return NULL;
	} else if (PyArray_NDIM(weighted_avg) != 1 || PyArray_TYPE(weighted_avg) != NPY_DOUBLE) {
		PyErr_Format(PyExc_ValueError, "Because weighted_avg array is %d-dimensional or not of type double", PyArray_NDIM(weighted_avg));
		return NULL;
	} else if (PyList_Size(beta_risks) != PyArray_DIM(timeslots, 0)) {
		PyErr_Format(PyExc_ValueError, "Because beta risk array is not the same length as the timeslots array.");
		return NULL;
	}

	// Now convert the non-lists to C-types
	beta = PyFloat_AS_DOUBLE(pybeta);

	// Actual algorithm follows
	result = 0;
	slotmax = PyArray_DIM(timeslots,0);
	
	for (slot_index = 0; slot_index < slotmax; slot_index++) {
		es = *(long *) PyArray_GETPTR1(timeslots, slot_index);
		output = *(double *) PyArray_GETPTR2(outputs, es, 0);
		z = (double *) PyArray_GETPTR1(part_func, slot_index);
		w = (double *) PyArray_GETPTR1(weighted_avg, slot_index);

		risk_group = (PyArrayObject*) PyList_GetItem(risk_groups, slot_index);
		riskmax = PyArray_DIM(risk_group, 0);

		beta_risk_outputs = (PyArrayObject*) PyList_GetItem(beta_risks, slot_index);

		*z = 0;
		*w = 0;
		
		for (risk_index = 0; risk_index < riskmax; risk_index++) {
			risk_output_index = *(long *) PyArray_GETPTR1(risk_group, risk_index);
			risk_output = *(double *) PyArray_GETPTR2(outputs, risk_output_index, 0);
			beta_risk = (double *) PyArray_GETPTR1(beta_risk_outputs, risk_index);
			
			*beta_risk = exp(beta * risk_output);
			*z += *beta_risk;
			*w += *beta_risk * risk_output;
		}
		*w /= *z;
		
		result += output - *w;
	}

	return Py_BuildValue("d", result); // Build the python type out of the C-result
};

// Method names that are importable from python
static PyMethodDef methods[] = {
	{"derivative_beta", // Function name, as seen from python
	derivative_beta, // actual C-function name
	METH_VARARGS, // positional (no keyword) arguments
	NULL}, // doc string for function
	{"get_slope", get_slope, METH_VARARGS, NULL},
};


// This bit is needed to be able to import from python

PyMODINIT_FUNC initcox_error_in_c()
{
	Py_InitModule("cox_error_in_c", methods);
}
