#include <Python.h>
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "structmember.h" // used to declare member list
#include <string.h> // need to compare names
#include <stdlib.h> // required for random()
#include "activation_functions.h"

#define TRUE 1
#define FALSE 0

/**
Node class
*/
typedef struct {
	PyObject_HEAD // Inherits from object
	double random_range;
	double bias;
	PyObject *weights; // A dictionary of nodes and weight values
	PyStringObject *activation_function; // The string representation of the activation function!
	double (*function)(double); // Function pointer, the activation function
	double (*derivative)(double); // Function pointer, to the derivative of the activation function
} Node;

static int _Node_input_sum(Node *self, PyObject *inputs, double *sum);
static int _Node_output(Node *self, PyObject *inputs, double *val);

/**
Public members
*/
static PyMemberDef NodeMembers[] = {
	{"bias", T_DOUBLE, offsetof(Node, bias), 0,
	 "Bias value"},
	{"weights", T_OBJECT_EX, offsetof(Node, weights), 0,
	 "The weights dict of {node, weight}"},
	{"activation_function", T_OBJECT_EX, offsetof(Node, activation_function), 0,
	 "String representation of the activation function (its name)"},
	{"random_range", T_DOUBLE, offsetof(Node, random_range), 0,
	 "Range within the weights are randomly assigned"},
	{NULL} // for safe iteration
};

/**
Safety method
Checks internal pointers for NULL and makes sure the weights dict has items in it.
*/
static int VerifySelf(Node *self)
{
	return (self->weights != NULL &&
		self->function != NULL &&
		self->derivative != NULL &&
		PyDict_Size(self->weights) > 0);
}

/**
Verifies that the input argument is a python list or a numpy list 1-D double list.
*/
static int VerifyList(PyObject *input)
{
	//printf("List: %d, NDIM = %d, NPY_DOUBLE: %d\n", PyList_CheckExact(input), PyArray_NDIM(input), PyArray_TYPE(input) == NPY_DOUBLE);
	return (PyList_CheckExact(input) || (PyArray_NDIM(input) == 1 && PyArray_TYPE(input) == NPY_DOUBLE));
}

/**
Constructor//Initializer//Destructor.
*/
static PyObject *
Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	Node *self;
	PyObject *weights = NULL;

	self = (Node *)type->tp_alloc(type, 0);
	if (self != NULL) {
		// Default values
		self->weights = NULL;
		self->random_range = 1;
		self->activation_function = NULL;

		double bias = RAND_MAX; // Dummy value that will never occur in real life

		static char *kwlist[] = {"active", "bias", "random_range", "weights", NULL};

		if (! PyArg_ParseTupleAndKeywords(args, kwds, "|SddO", kwlist,
							&self->activation_function, &bias,
							&self->random_range,
							&weights))
		{
			PyErr_Format(PyExc_ValueError, "Arguments should be (all optional): string active, double bias, double random_range, dict weights");
			return NULL;
		}
		
		// Set bias
		if (bias != RAND_MAX) {
			self->bias = bias;
		}
		else {
	 		// Assign random value based on random range
	 		srand((unsigned)time(NULL)); // Seed it
			self->bias = self->random_range * ((double)rand()/(double)RAND_MAX);
		}

		// Weights
		if (weights == NULL) 
		{
			self->weights = PyDict_New();
		}
		else if (PyDict_Check(weights)) {
			self->weights = weights;
			Py_INCREF(self->weights);
		}
		else {
			// Incorrect object...
			PyErr_Format(PyExc_ValueError, "Weights was not a dict!");
			return NULL;
		}

		// Set activation function and derivative
		if (self->activation_function != NULL && strcmp (PyString_AS_STRING(self->activation_function), "logsig" ) == 0)
		{
			Py_INCREF(self->activation_function);
			self->function = logsig;
			self->derivative = logsig_derivative;
		}
		else if (self->activation_function != NULL && strcmp (PyString_AS_STRING(self->activation_function), "tanh" ) == 0)
		{
			Py_INCREF(self->activation_function);
			self->function = tanh;
			self->derivative = tanh_derivative;
		}
		else // Linear it is!
		{
		self->activation_function = (PyStringObject*) PyString_FromString("linear");
		self->function = linear;
		self->derivative = linear_derivative;
		}
	} //if !self null

	return (PyObject *)self;
}
static int
Node_init(Node *self, PyObject *args, PyObject *kwds)
{
	return 0;
}

static void
Node_dealloc(Node *self)
{
	Py_XDECREF(self->weights);
	Py_XDECREF(self->activation_function);
	self->ob_type->tp_free((PyObject*)self);
}

/**
Node Class methods
Returns an error code. TRUE for success, FALSE for error...
*/
static int _Node_input_sum(Node *self, PyObject *inputs, double *sum)
{
	// Iterate over the items in the weights dict

	PyObject *key, *pyweight;
	double weight, value = 0, recursive_val;
	Py_ssize_t pos = 0;
	int result = TRUE;
	*sum = self->bias;

	while (PyDict_Next(self->weights, &pos, &key, &pyweight)) {
		// First verify that the weight is a float
		if (!PyFloat_Check(pyweight)) {
			PyErr_Format(PyExc_ValueError, "The value of the weights dict was NOT a float!");
	 		result = FALSE;
		}
		else {
			weight = PyFloat_AS_DOUBLE(pyweight);
			// If node is a Node, call its inputsum. Else, it's an input index.
			if (PyObject_IsInstance(key, (PyObject*) self->ob_type))
			{

				if (_Node_output((Node*) key, inputs, &recursive_val))
					*sum += weight * recursive_val;
				else
					result = FALSE;
			}
			else if (PyInt_Check(key))// It's an input index
			{
				Py_ssize_t index = PyInt_AsSsize_t(key);
				// Two cases, python list or numpy list
				if (PyList_CheckExact(inputs)) // Python list
				{
					PyObject *pyval = PyList_GetItem(inputs, index);
					if (pyval == NULL)
						result = FALSE;
					else
			 			value = PyFloat_AS_DOUBLE(pyval);
				}
				else // Numpy list
				{
					double *ptr = (double *) PyArray_GETPTR1((PyArrayObject*) inputs, index);
					if (ptr == NULL)
						result = FALSE;
					else
			 			value = *ptr;
				}

				*sum += weight * value;
			}
			else // Someone fucked up the weight dict
			{
				PyErr_Format(PyExc_ValueError, "The key of the weights dict was neither a Node nor an Int!");
			 	result = FALSE;
			}
		} //if float
	} //while
	return result;
}

static PyObject* Node_output_derivative(Node *self, PyObject *inputs)
{
	if (!VerifyList(inputs))
	{
		PyErr_Format(PyExc_ValueError, "The input is not a one dimension numpy list of type Double, or not a python list.");
		return NULL;
	}
	else if (!VerifySelf(self))
	{
		PyErr_Format(PyExc_ValueError, "The weights vector is empty or an internal pointer is NULL: %d (1 for yes)", (self->weights == NULL || self->function == NULL || self->derivative == NULL));
		return NULL;
	}
	else
	{
		double inputsum;
		if(_Node_input_sum(self, inputs, &inputsum))
			return Py_BuildValue("d", self->derivative(inputsum));
		else
			return NULL;
	}
}

static int _Node_output(Node *self, PyObject *inputs, double *val)
{
	int result = TRUE;
	double inputsum;
	if(_Node_input_sum(self, inputs, &inputsum))
		*val = self->function(inputsum);
	else
		result = FALSE;
	
	return result;
}
static PyObject* Node_output(Node *self, PyObject *inputs)
{
	if (!VerifyList(inputs))
	{
		PyErr_Format(PyExc_ValueError, "The input is not a one dimension numpy list of type Double, or not a python list.");
		return NULL;
	}
	else if (!VerifySelf(self))
	{
		PyErr_Format(PyExc_ValueError, "The weights vector is empty or an internal pointer is NULL: %d (1 for yes)", (self->weights == NULL || self->function == NULL || self->derivative == NULL));
		return NULL;
	}
	else
	{
	double val;
	if (_Node_output(self, inputs, &val)) {
		return Py_BuildValue("d", val);
	}
	else
		return NULL;
	}
}

/**
Used in pickling
Returns the arguments necessary to reconstruct this object.
*/
static PyObject* Node_getnewargs(Node* self)
{
	//"active", "bias", "random_range", "weights"
	return Py_BuildValue("(SddO)", self->activation_function, self->bias, self->random_range, self->weights);
}

/**
This method is responsible for telling the pickler how to reconstruct this object.
It returns a constructer (Py_Type(self)) and the arguments that accepts to reconstruct this object.
*/
static PyObject* Node_reduce(Node* self)
{
	PyObject *args = Node_getnewargs(self);
	if (args == NULL)
		return NULL; // Error, an exception should have been set
	return Py_BuildValue("(OO)", Py_TYPE(self), args);
}

/**
Specify the accessible methods in a list
*/

static PyMethodDef NodeMethods[] = 
{
	{"output_derivative", (PyCFunction) Node_output_derivative, METH_O, "The derivative of the activation function, given the inputs"},
	{"output", (PyCFunction) Node_output, METH_O, "The derivative of the activation function, given the inputs"},
	{"__reduce__", (PyCFunction) Node_reduce, METH_NOARGS, "Needed for pickling. Specifices how to reconstruct the object."},
	{"__getnewargs__", (PyCFunction) Node_getnewargs, METH_NOARGS, "Needed for pickling. Specifices what args to give new()."},
	{NULL}, // So that we can iterate safely below
};

/**
Specify the type of this class
*/

static PyTypeObject
NodeType = {
	PyObject_HEAD_INIT(NULL)
	0,						/* ob_size */
	"kalderstam.neural.fast_network.Node",		/* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
	sizeof(Node),					/* tp_basicsize */
	0,						/* tp_itemsize */
	(destructor)Node_dealloc,			/* tp_dealloc */
	0,						/* tp_print */
	0,						/* tp_getattr */
	0,						/* tp_setattr */
	0,						/* tp_compare */
	0,						/* tp_repr */
	0,						/* tp_as_number */
	0,						/* tp_as_sequence */
	0,						/* tp_as_mapping */
	0,						/* tp_hash */
	0,						/* tp_call */
	0,						/* tp_str */
	0,						/* tp_getattro */
	0,						/* tp_setattro */
	0,						/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, 	/* tp_flags*/
	"A node in a neural network.",			/* tp_doc */
	0,						/* tp_traverse */
	0,			 			/* tp_clear */
	0,			 			/* tp_richcompare */
	0,			 			/* tp_weaklistoffset */
	0,			 			/* tp_iter */
	0,			 			/* tp_iternext */
	NodeMethods,					/* tp_methods */
	NodeMembers,					/* tp_members */
	0,			 			/* tp_getset */
	0,			 			/* tp_base */
	0,			 			/* tp_dict */
	0,			 			/* tp_descr_get */
	0,			 			/* tp_descr_set */
	0,			 			/* tp_dictoffset */
	(initproc)Node_init,				/* tp_init */
	0,			 			/* tp_alloc */
	Node_new,			 		/* tp_new */
};


void
initfast_network(void)
{
	PyObject* mod;

	// Create the module
	mod = Py_InitModule3("fast_network", NULL, "C implementation of the neural network node.");
	if (mod == NULL) {
		return;
	}

	// Make it ready
	if (PyType_Ready(&NodeType) < 0) {
		return;
	}

	// Add the type to the module.
	Py_INCREF(&NodeType);
	PyModule_AddObject(mod, "Node", (PyObject*)&NodeType);
}
