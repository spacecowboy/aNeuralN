#include <Python.h>
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "structmember.h" // used to declare member list

#include "fast_node.h"

/**
Network class
*/
typedef struct {
  PyObject_HEAD // Inherits from object
  Py_ssize_t num_of_inputs;
  BiasNode *bias_node;
  PyObject *hidden_nodes; // List
  PyObject *output_nodes; // List
} Network;

//static int Network_len(Network *self);
//static int _Network_predict(Network *self, PyObject *input_array);
//static int _Network_fit(Network *self, PyObject *input_array, PyObject *target_array);
//static int _Network_score(Network *self, PyObject *input_array);

/**
Public members
*/
static PyMemberDef NetworkMembers[] = {
  {"num_of_inputs", T_INT, offsetof(Network, num_of_inputs), 0,
   "The number of features this network expects in input data"},
  {"bias_node", T_OBJECT_EX, offsetof(Network, bias_node), 0,
   "A network has only one bias node, connected or not."},
  {"hidden_nodes", T_OBJECT_EX, offsetof(Network, hidden_nodes), 0,
   "A list of the hidden nodes in this network. Potentially empty."},
  {"output_nodes", T_OBJECT_EX, offsetof(Network, output_nodes), 0,
   "A list of the output nodes in this network. Usually contains only 1 node."},
  {NULL} // for safe iteration
};

/**
Constructor
*/
static PyObject *
Network_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  Network *self;
  self = (Network *)type->tp_alloc(type, 0);
  return (PyObject *)self;
}

/**
Init takes no arguments
*/
static int
Network_init(Network *self, PyObject *args, PyObject *kwargs)
{
  self->num_of_inputs = 0;
  self->bias_node = (BiasNode*) PyObject_CallObject((PyObject *) &BiasNodeType, NULL);
  self->hidden_nodes = PyList_New(0);
  self->output_nodes = PyList_New(0);

  // Add handling of arguments here to support pickling
  return 0;
}

/**
Destructor, decrement memory count for our objects
*/
static void
Network_dealloc(Network *self)
{
  Py_XDECREF(self->hidden_nodes);
  Py_XDECREF(self->output_nodes);
  Py_XDECREF(self->bias_node);
  self->ob_type->tp_free((PyObject*)self);
}

/**
Network class methods
*/

/**
The length of a network is defined as input nodes + hidden nodes + output nodes
*/
static PyObject* Network_len(Network *self)
{
  Py_ssize_t length = self->num_of_inputs;
  length += PyList_Size(self->hidden_nodes);
  length += PyList_Size(self->output_nodes);

  // Python integer
  return Py_BuildValue("n", length);
}

static PyObject* Network_predict(Network *self, PyObject *input_array)
{
  if (!(PyList_CheckExact(input_array) || PyArray_NDIM(input_array) == 1)) {
    PyErr_Format(PyExc_ValueError, "The input is a one dimension numpy list, or not a python list.");
    return NULL;
  }
  // First verify that we have output nodes
  Py_ssize_t num_of_outputs = PyList_Size(self->output_nodes);
  int num_of_dims;
  if (num_of_outputs < 1) {
    PyErr_Format(PyExc_ValueError, "This network has no output nodes!");
    return NULL;
  } else {
    num_of_dims = 2;
  }
  //First verify that input_array is iterable

  // Then iterate
  Py_ssize_t length = 0;
  Py_ssize_t index;
  PyObject *inputs;
  int numpy_list = 0;
  // Two cases, python list or numpy list
  if (PyList_CheckExact(input_array)) {
    length = PyList_Size(input_array);
    numpy_list = 0;
  } else {
    length = PyArray_Size(input_array);
    numpy_list = 1;
  }

  // Create return array
  npy_intp dims[num_of_dims];
  dims[0] = length;
  //if (num_of_dims > 1) {
  dims[1] = num_of_outputs;
  //}
  PyArrayObject *result_array = (PyArrayObject *) PyArray_SimpleNew(num_of_dims, dims, NPY_DOUBLE);

  if (result_array == NULL) {
    PyErr_Format(PyExc_ValueError, "Failed to allocate the result array");
    return NULL;
  }

  // Loop over the data
  Py_ssize_t node_index;
  Node *node;
  double output;
  double *result_i_j;

  for (index = 0;index < length; index++) {
    if (numpy_list) {
      inputs = PyArray_GETPTR1((PyArrayObject*) input_array, index);
    }
    else {
      inputs = PyList_GetItem(input_array, index);
    }
    // Loop over output nodes
    for (node_index = 0; node_index < num_of_outputs; node_index++) {
      node = (Node *) PyList_GetItem(self->output_nodes, node_index);
      if (_Node_output(node, inputs, &output)) {
        result_i_j = (double *) PyArray_GETPTR2(result_array, index, node_index);
        *result_i_j = output;
      }
    }
  }

  // result array done. Squeeze it to get rid of unnecessary dimensions
  return PyArray_Squeeze(result_array);
}

//static int _Network_output(Network *self)
//{
//}

//static int _Network_fit(Network *self, PyObject *input_array, PyObject *target_array)
//{
//}

//static int _Network_score(Network *self, PyObject *input_array)
//{
//}


/**
Used in pickling
Returns the arguments necessary to reconstruct this object.
*/
static PyObject* Network_getnewargs(Network* self)
{
  // Arguments match the struct of network
  return Py_BuildValue("(nOOO)", self->num_of_inputs, self->bias_node, self->hidden_nodes, self->output_nodes);
}

/**
This method is responsible for telling the pickler how to reconstruct this object.
It returns a constructer (Py_Type(self)) and the arguments that accepts to reconstruct this object.
*/
static PyObject* Network_reduce(Network* self)
{
	PyObject *args = Network_getnewargs(self);
	if (args == NULL)
		return NULL; // Error, an exception should have been set
	return Py_BuildValue("(OO)", Py_TYPE(self), args);
}


/**
Python boiler plate
*/

/**
Specify the accessible methods
*/

static PyMethodDef NetworkMethods[] =
{
	{"__len__", (PyCFunction) Network_len, METH_NOARGS, "Length = hidden nodes + output nodes"},
	{"predict", (PyCFunction) Network_predict, METH_O, "Predicts values for each set of inputs in the array."},
	{"__reduce__", (PyCFunction) Network_reduce, METH_NOARGS, "Needed for pickling. Specifices how to reconstruct the object."},
	{"__getnewargs__", (PyCFunction) Network_getnewargs, METH_NOARGS, "Needed for pickling. Specifices what args to give new()."},
	{NULL}, // So that we can iterate safely below
};

/**
Specify the type of this class
*/

static PyTypeObject
NetworkType = {
	PyObject_HEAD_INIT(NULL)
	0,						/* ob_size */
	"ann.network.Network",		/* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
	sizeof(Network),					/* tp_basicsize */
	0,						/* tp_itemsize */
	(destructor)Network_dealloc,			/* tp_dealloc */
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
	"A neural network.",			/* tp_doc */
	0,						/* tp_traverse */
	0,			 			/* tp_clear */
	0,			 			/* tp_richcompare */
	0,			 			/* tp_weaklistoffset */
	0,			 			/* tp_iter */
	0,			 			/* tp_iternext */
	NetworkMethods,					/* tp_methods */
	NetworkMembers,					/* tp_members */
	0,			 			/* tp_getset */
	0,			 			/* tp_base */
	0,			 			/* tp_dict */
	0,			 			/* tp_descr_get */
	0,			 			/* tp_descr_set */
	0,			 			/* tp_dictoffset */
	(initproc)Network_init,				/* tp_init */
	0,			 			/* tp_alloc */
	Network_new,			 		/* tp_new */
};


/* Module wide stuff */

void
initfast_network(void)
{
	PyObject* mod;

	// Create the module
	mod = Py_InitModule3("fast_network", NULL, "C implementation of the neural network.");
	if (mod == NULL) {
		return;
	}

	// Make it ready
	if (PyType_Ready(&NetworkType) < 0) {
		return;
	}

	// Add the type to the module.
	Py_INCREF(&NetworkType);
	PyModule_AddObject(mod, "Network", (PyObject*)&NetworkType);
}
