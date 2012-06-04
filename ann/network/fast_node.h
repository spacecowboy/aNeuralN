#define TRUE 1
#define FALSE 0

/**
Node class
*/
typedef struct {
	PyObject_HEAD // Inherits from object
	double random_range;
	PyObject *weights; // A dictionary of nodes and weight values
	PyStringObject *activation_function; // The string representation of the activation function!
	double (*function)(double); // Function pointer, the activation function
	double (*derivative)(double); // Function pointer, to the derivative of the activation function
	double cached_output;
	PyObject *cached_input;
} Node;

static PyMemberDef NodeMembers[];
static PyMethodDef NodeMethods[];

static int _Node_output(Node *self, PyObject *inputs, double *output);

/**
Bias Node class
*/
typedef struct {
	Node node; // Inherits from Node
} BiasNode;

/**
Specify the type of this class
*/

static PyMethodDef BiasNodeMethods[];


static PyObject *BiasNode_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int BiasNode_init(BiasNode *self, PyObject *args, PyObject *kwds);


static PyTypeObject
BiasNodeType = {
	PyObject_HEAD_INIT(NULL)
	0,						/* ob_size */
	"ann.network.BiasNode",	/* tp_name */ // VITAL THAT THIS IS CORRECT PACKAGE NAME FOR PICKLING!
	sizeof(BiasNode),				/* tp_basicsize */
	0,						/* tp_itemsize */
	0,						/* tp_dealloc */
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
	"A bias node in a neural network.",		/* tp_doc */
	0,						/* tp_traverse */
	0,			 			/* tp_clear */
	0,			 			/* tp_richcompare */
	0,			 			/* tp_weaklistoffset */
	0,			 			/* tp_iter */
	0,			 			/* tp_iternext */
	BiasNodeMethods,				/* tp_methods */
	NodeMembers,						/* tp_members */
	0,			 			/* tp_getset */
	0,			 			/* tp_base */
	0,			 			/* tp_dict */
	0,			 			/* tp_descr_get */
	0,			 			/* tp_descr_set */
	0,			 			/* tp_dictoffset */
	(initproc)BiasNode_init,				/* tp_init */
	0,			 			/* tp_alloc */
	BiasNode_new,			 		/* tp_new */
};
