#include <Python.h>
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "structmember.h" // used to declare member list
#include <string.h> // need to compare names
#include <stdlib.h> // required for random()
#include "activation_functions.h"

/**
Node class
*/
typedef struct {
   PyObject_HEAD // Inherits from object
   double random_range;
   double bias;
   PyObject *weights; // A dictionary of nodes and weight values
   char *activation_function; // The string representation of the activation function!
   double (*function)(double); // Function pointer, the activation function
   double (*derivative)(double); // Function pointer, to the derivative of the activation function
} Node;

static double _Node_input_sum(Node *self, PyObject *inputs);
static double _Node_output(Node *self, PyObject *inputs);

/**
Public members
*/
static PyMemberDef NodeMembers[] = {
    {"bias", T_DOUBLE, offsetof(Node, bias), 0,
     "Bias value"},
    {"weights", T_OBJECT_EX, offsetof(Node, weights), 0,
     "The weights dict of {node, weight}"},
    {"activation_function", T_STRING, offsetof(Node, activation_function), 0,
     "String representation of the activation function (its name)"},
    {"random_range", T_DOUBLE, offsetof(Node, random_range), 0,
     "Range within the weights are randomly assigned"},
    {NULL}  // for safe iteration
};

/**
Constructor//Initializer//Destructor.
*/
static PyObject *
Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Node *self;
    printf("New called!\n");

    self = (Node *)type->tp_alloc(type, 0);
    if (self != NULL) {

	// Default values
	   self->weights = NULL;
	   self->random_range = 1;
	   self->activation_function = "linear";

	   double bias = RAND_MAX; // Dummy value that will never occur in real life

	   static char *kwlist[] = {"active", "bias", "random_range", "weights", NULL};

	   if (! PyArg_ParseTupleAndKeywords(args, kwds, "|sddO", kwlist,
		                              &self->activation_function, &bias,
		                              &self->random_range,
					      &self->weights))
	       {
		PyErr_Format(PyExc_ValueError, "Arguments should be: string active, double bias, double random_range");
		return NULL;
		}
	   printf("And the function is: %s\n", self->activation_function);
        
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
	   if (self->weights == NULL) 
	   {
		self->weights = PyDict_New();
	   }
	   else {
		Py_INCREF(&self->weights);
	   }

	   // Set activation function and derivative
	   if (strcmp (self->activation_function, "logsig" ) == 0)
	   {
	       self->function = logsig;
	       self->derivative = logsig_derivative;
	   }
	   else if (strcmp (self->activation_function, "tanh" ) == 0)
	   {
	       self->function = tanh;
	       self->derivative = tanh_derivative;
	   }
	   else // Linear it is!
	   {
	       self->activation_function = "linear";
	       self->function = linear;
	       self->derivative = linear_derivative;
	   }
    }

    return (PyObject *)self;
}
static int
Node_init(Node *self, PyObject *args, PyObject *kwds)
{
   printf("Init called!\n");
   // Default values
/*
   self->weights = PyDict_New();
   self->random_range = 1;
   self->activation_function = "linear";

   double bias = RAND_MAX; // Dummy value that will never occur in real life

   static char *kwlist[] = {"active", "bias", "random_range", NULL};

   if (! PyArg_ParseTupleAndKeywords(args, kwds, "|sdd", kwlist,
                                      &self->activation_function, &bias,
                                      &self->random_range))
       return -1;

   // Set bias
   if (bias != RAND_MAX) {
       self->bias = bias;
   }
   else {
       // Assign random value based on random range
       srand((unsigned)time(NULL)); // Seed it
       self->bias = self->random_range * ((double)rand()/(double)RAND_MAX);
   }

   // Set activation function and derivative
   if (strcmp (self->activation_function, "logsig" ) == 0)
   {
       self->function = logsig;
       self->derivative = logsig_derivative;
   }
   else if (strcmp (self->activation_function, "tanh" ) == 0)
   {
       self->function = tanh;
       self->derivative = tanh_derivative;
   }
   else // Linear it is!
   {
       self->activation_function = "linear";
       self->function = linear;
       self->derivative = linear_derivative;
   }
*/
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
These are the only ones that need speed, rest are implemented in python.
*/
static double _Node_input_sum(Node *self, PyObject *inputs)
{
    // Iterate over the items in the weights dict

    PyObject *key, *pyweight;
    double weight, value, sum = self->bias;
    Py_ssize_t pos = 0;

    while (PyDict_Next(self->weights, &pos, &key, &pyweight)) {
        weight = PyFloat_AS_DOUBLE(pyweight);
        // If node is a Node, call its inputsum. Else, it's an input index.
        if (PyObject_IsInstance(key, (PyObject*) self->ob_type))
        {
             sum += weight * _Node_output((Node*) key, inputs);
        }
        else // It's an input index
        {
             Py_ssize_t index = PyInt_AsSsize_t(key);
	     // Two cases, python list or numpy list
             if (PyList_CheckExact(inputs))
	     {
             	PyObject *pyval = PyList_GetItem(inputs, index);
	     	value = PyFloat_AS_DOUBLE(pyval);
	     }
	     else // Numpy list
	     {
	     	value = *(double *) PyArray_GETPTR1((PyArrayObject*) inputs, index);
	     }

	     sum += weight * value;
        }    
    }

    return sum;
}
static PyObject* Node_input_sum(Node *self, PyObject *inputs)
{
    printf("C values = %p, %p, %g, %g, %s\n", self->function, self->derivative, self->bias, self->random_range, self->activation_function);
    if (self->function == NULL || self->derivative == NULL)
    {
         PyErr_Format(PyExc_ValueError, "Something was null! function=%p, derivative=%p", self->function, self->derivative);
	 return NULL;
    }
    else
    {
    	double sum = _Node_input_sum(self, inputs);

    	return Py_BuildValue("d", sum);
    }
}

static double _Node_output_derivative(Node *self, PyObject *inputs)
{
    double inputsum = _Node_input_sum(self, inputs);
    return self->derivative(inputsum);
}
static PyObject* Node_output_derivative(Node *self, PyObject *inputs)
{
    double val = _Node_output_derivative(self, inputs);

    return Py_BuildValue("d", val);
}

static double _Node_output(Node *self, PyObject *inputs)
{
    double inputsum = _Node_input_sum(self, inputs);
    return self->function(inputsum);
}
static PyObject* Node_output(Node *self, PyObject *inputs)
{
    double val = _Node_output(self, inputs);

    return Py_BuildValue("d", val);
}

/**
Used in pickling
*/
static PyObject* Node_getnewargs(Node* self)
{
	printf("GETNEWARGS! %s\n", self->activation_function);
	//"active", "bias", "random_range", "weights"
	return Py_BuildValue("(sddO)", self->activation_function, self->bias, self->random_range, self->weights);
}

static PyObject* Node_reduce(Node* self)
{
	printf("REDUCE!\n");
	//"active", "bias", "random_range", "weights"
	PyObject *args = Node_getnewargs(self);
	if (args == NULL)
		return NULL; // Error, an exception should have been set
	printf("Got args\n");
	//return Py_BuildValue("(OO)", Py_TYPE(self), args);
	// Retrieve the class object
	PyObject *attr_name = Py_BuildValue("s", "__class__");
	PyObject *class = PyObject_GetAttr(self, attr_name);
	return Py_BuildValue("(OO)", class, args);
}

/**
Specify the accessible methods in a list
*/

static PyMethodDef NodeMethods[] = 
{
    {"input_sum", (PyCFunction) Node_input_sum, METH_O, "The sum of the inputs to this Node"},
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
   0,                         /* ob_size */
   "kalderstam.neural.fast_network.Node",               /* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
   sizeof(Node),         /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)Node_dealloc, /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_compare */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
   "A node in a neural network.",              /* tp_doc */
   0,                         /* tp_traverse */
   0,                         /* tp_clear */
   0,                         /* tp_richcompare */
   0,                         /* tp_weaklistoffset */
   0,                         /* tp_iter */
   0,                         /* tp_iternext */
   NodeMethods,         /* tp_methods */
   NodeMembers,         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Node_init,  /* tp_init */
   0,                         /* tp_alloc */
   Node_new,                         /* tp_new */
};


void
initfast_network(void)
{
   PyObject* mod;

   // Create the module
   mod = Py_InitModule3("fast_network", NULL, "Base C implementation of the neural network node.");
   if (mod == NULL) {
      return;
   }

   // Fill in some slots in the type, and make it ready
   //NodeType.tp_new = PyType_GenericNew;
   //NodeType.tp_base = &PyType_Type;
   if (PyType_Ready(&NodeType) < 0) {
      return;
   }

   // Add the type to the module.
   Py_INCREF(&NodeType);
   PyModule_AddObject(mod, "Node", (PyObject*)&NodeType);
}
