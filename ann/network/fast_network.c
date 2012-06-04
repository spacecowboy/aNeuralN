#include <Python.h>
#include <numpy/arrayobject.h> // NumPy as seen from C
#include "structmember.h" // used to declare member list

#include "fast_node.h"

/**
Network class
*/
typedef struct {
  PyObject_HEAD // Inherits from object
  int num_of_inputs;
  Node *bias_node;
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
Constructors, destructors etc
*/

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
  self->bias_node = (Node*) PyObject_CallObject((PyObject *) &BiasNodeType, NULL);
  self->hidden_nodes = PyList_New();
  self->output_nodes = PyList_New();
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
  int result = TRUE;
  Py_ssize_t length = self->num_of_inputs;
  length += PyList_Size(self->hidden_nodes);
  length += PyList_Size(self->output_nodes);

  // Python integer
  return Py_BuildValue("n", length);
}

/**

 */
static int _Network_predict(Network *self, PyObject *input_array)
{
  // First verify that we have output nodes
  //First verify that input_array is iterable
  // Then iterate
  // For every item, verify that the item is iterable and of correct length (num_of_inputs)
  // Then let every output node give an answer, store them
  // Build and return numpy array of predictions
  // If only one output node, dimension (rows,) else (rows, nodes)
}

//static int _Network_output(Network *self)
//{
//}

static int _Network_fit(Network *self, PyObject *input_array, PyObject *target_array)
{
}

static int _Network_score(Network *self, PyObject *input_array)
{
}
