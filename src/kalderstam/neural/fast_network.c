#include <Python.h>

/**
Node class
*/
typedef struct {
   PyObject_HEAD // Inherits from object
   double random_range;
   double bias;
   PyObject *weights; // A dictionary of nodes and weight values
   PyObject *activation_function; // This should ideally have a C-API
} Node;

/**
Public members (accessible from Python). Don't know why, but this fails to compile, using getters/setters instead.
static PyMemberDef NodeMembers[] = {
    {"random_range", T_DOUBLE, offsetof(Node, random_range), 0,
     "Range within a weight or bias is randomized from."},
    {"bias", T_DOUBLE, offsetof(Node, bias), 0,
     "Bias value"},
    {"weights", T_OBJECT_EX, offsetof(Node, weights), 0,
     "Dict of {Node, weight}"},
    {"activation_function", T_OBJECT_EX, offsetof(Node, activation_function), 0,
     "The activation function."},
    {NULL}  // Sentinel, for safe iterating
};
*/

/**
Getters and Setters, finishing with a list of all methods
*/

static PyObject *
Node_getrandom_range(Node *self, void *closure)
{
    return Py_BuildValue("d", self->random_range);
}

static int
Node_setrandom_range(Node *self, PyObject *value, void *closure)
{
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the random range attribute");
    return -1;
  }
  
  if (! PyFloat_Check(value)) {
    PyErr_SetString(PyExc_TypeError, 
                    "The random range attribute value must be a float");
    return -1;
  }

  self->random_range = PyFloat_AS_DOUBLE(value);    

  return 0;
}


static PyGetSetDef Node_getseters[] = {
    {"random_range", 
     (getter)Node_getrandom_range, (setter)Node_setrandom_range,
     "Range within a weight or bias is randomized from",
     NULL},
    {"bias", 
     (getter)Node_getbias, (setter)Node_setbias,
     "Bias value",
     NULL},
    {"weights", 
     (getter)Node_getweights, (setter)Node_setweights,
     "Dict of {Node, weight}",
     NULL},
    {"activation_function", 
     (getter)Node_getactivation_function, (setter)Node_setactivation_function,
     "The activation function",
     NULL},
    {NULL}  /* Sentinel */
};

/**
Constructor/Destructor.
*/
static int
Node_init(Node *self, PyObject *args, PyObject *kwds)
{
   self->weights = PyDict_New();
   //self->activation_function == NULL // Should assign this...
   self->bias = 0;

   return 0;
}

static void
Node_dealloc(Node *self)
{
   Py_XDECREF(self->weights);
   self->ob_type->tp_free((PyObject*)self);
}

/**
Node Class methods
These are the only ones that need speed, rest are implemented in python.
*/
static PyObject* Node_input_sum(PyObject *self, PyObject *args)
{
    printf("The fire pops and sizzles in C...\n");

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* Node_output_derivative(PyObject *self, PyObject *args)
{
    printf("KABOOOOM! Exploded in C!\n");

    Py_INCREF(Py_None);
    return Py_None;
}

/**
Specify the accessible methods in a list
*/

static PyMethodDef NodeMethods[] = 
{
    {"input_sum", (PyCFunction) Node_input_sum, METH_VARARGS, "The sum of the inputs to this Node"},
    {"output_derivative", (PyCFunction) Node_output_derivative, METH_VARARGS, "The derivative of the activation function, given the inputs"},
    {NULL}, // So that we can iterate safely below
};

/**
Specify the type of this class
*/

static PyTypeObject
NodeType = {
   PyObject_HEAD_INIT(NULL)
   0,                         /* ob_size */
   "Node",               /* tp_name */
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
   0,                         /* tp_new */
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
   NodeType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&NodeType) < 0) {
      return;
   }

   // Add the type to the module.
   Py_INCREF(&NodeType);
   PyModule_AddObject(mod, "Node", (PyObject*)&NodeType);
}
