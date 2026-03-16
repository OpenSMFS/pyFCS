#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

typedef enum
{
	CC_UNDEFINED = 0,
	CC_TIMES = 1,
	CC_WEIGHTSLIST = 2,
	CC_NANOS = 3,
	CC_WEIGHTSINDEX = 4
	
} CC_dtype;

typedef enum
{
	CastAny,
	CastInteger
} CC_CastSel;

typedef struct
{
	size_t n;
	size_t *lens;
	cc_times **times;
} CCtimeslist;

typedef struct
{
	size_t n;
	size_t nW;
	size_t *lens;
	cc_weights ***weights;
} CCweightslist;

typedef struct
{
	size_t n;
	size_t *lens;
	cc_nanos **times;
} CCnanoslist;

typedef struct
{
	size_t nW;
	size_t nI;
	cc_weights **weights;
} CCweightsindex;

typedef union
{
	CCtimeslist *times;
	CCweightslist *weightsL;
	CCnanoslist *nanos;
	CCweightsindex *weightsI;
} CC_data;

typedef struct
{
	size_t n;
	PyArrayObject **npdata;
	CC_dtype dtype;
	CC_data cdata;
} CCnplist;

static inline PyArrayObject* npArray_FromAny(PyObject *in, int typenum)
{
	PyObject *cast = NULL;
	PyArrayObject *out = NULL;
	if (in != NULL)
	{
		cast = PyArray_FROM_OTF(in, typenum, NPY_ARRAY_CARRAY_RO);
		if ( cast != NULL)
		{
			if (PyArray_Check(cast))
				out = (PyArrayObject*) cast;
			else
				Py_DECREF(cast);
		}
	}
	return out;
}
	
static inline PyArrayObject* npArray_FromInteger(PyObject *in, int typenum)
{
	PyObject *precast = NULL;
	PyObject *cast = NULL;
	PyArrayObject *out = NULL;
	if ( in != NULL )
	{
		precast = PyArray_FROM_O(in);
		if (precast != NULL)
		{
			if (PyArray_Check(precast))
			{
				if (PyArray_ISINTEGER((PyArrayObject*) precast))
				{
					cast = PyArray_FROM_OTF(precast, typenum, NPY_ARRAY_CARRAY_RO|NPY_ARRAY_FORCECAST);
					Py_DECREF(precast);
					precast = NULL;
					if (cast != NULL)
					{
						if (PyArray_Check(cast))
						{
							out = (PyArrayObject*) cast;
							cast = NULL;
						}
						else
						{
							Py_DECREF(cast);
							cast = NULL;
						}
					}
				}
			}
			else
			{
				Py_DECREF(precast);
				precast = NULL;
			}
		}
	}
	return out;
}


static inline CCnplist* cast_sequence(PyObject *in, CC_CastSel castfunc, CC_dtype castsort, const char *name, const char *requirements)
{
	CCnplist *out = NULL;
	PyObject *check = NULL;
	PyObject *arrcheck = NULL;
	Py_ssize_t len_check = 0;
	if (PySequence_Check(in)
	{
		len_check = PySequence_Size(in);
		if (len_check < 0)
			PyErr_Format(PyExc_TypeError, "%s is not sequence, must be %s or sequence thereof", name, requirements);
		else if (len_check == 0)
			PyErr_Format(PyExc_TypeError, "%s is empty sequence, must be %s or sequence thereof", name, requirements);
		else
		{
			check = PySequence_GetItem(in, 0);
			if (check != NULL)
			{
				if (PySequence_Check(check))
				{
					arrcheck = 
				}
				else
				{
				}
				Py_DECREF(check);
			}
		}
	}
	return out;
}

static PyObject* pyFCS_testsequence(PyObject *self, PyObject *args, PyObject *kwargs)
{
	PyObject *pyTimes=NULL, *pyNanos=NULL;
	char *kwlist[] = {"times", "nanos", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &pyTimes, &pyNanos))
		goto exit;
	Py_ssize_t len = PyTuple_Size(pyTimes);
	if (len < 0)
		goto exit;
	PyObject *tup = PyTuple_New(len);
	for (Py_ssize_t i = 0; i < len; i++)
	{
		printf("getting object %ld\n", i);
		PyObject *tempo = PyTuple_GetItem(pyTimes, i);
		if (PyErr_Occurred() != NULL)
			printf("some error occured after PyTuple_GetItem\n");
		printf("pyTimes = %p, tempo = %p\n", pyTimes, tempo);
		PyObject *temp = npArray_FromInteger(tempo, NPY_UINT64);
		if (PyErr_Occurred() != NULL)
			printf("some error occured after PyArray_FROM_OT\n");
		printf("temp = %p\n", temp);
		if (temp == NULL)
		{
			printf("error occured in temp\n");
			PyErr_Format(PyExc_ValueError, "element %ld is not an array", i);
			Py_DECREF(tup);
			tup = NULL;
			break;
		}
		printf("setting tuple tup = %p\n", tup);
		if (PyTuple_SetItem(tup, i, temp)!=0)
		{
			printf("tuple not set\n");
			PyErr_Format(PyExc_ValueError, "cannot set element %ld", i);
			Py_DECREF(tup);
			tup = NULL;
			break;
		}
		if (PyErr_Occurred() != NULL)
			printf("some error occured after PyTuple_Pack\n");
		printf("set\n");
	}
	printf("done\n");
	exit:
	printf("returning tup = %p\n", tup);
	if (PyErr_Occurred() != NULL)
	{
		printf("exception set somewhere\n");
		Py_XDECREF(tup);
		tup = NULL;
	}
	return tup;
}

// Python correlate function
/*static PyObject *pyFCS_correlate(PyObject *self, PyObject *args, PyObject *kwargs)
{
	// Declaration of input arguments
	PyObject *pytimesT=NULL, *pytimesU=NULL, *pybins=NULL, *pyweightsT=NULL, *pyweightsU=NULL, *pynanosT=NULL, *pynanosU=NULL, *pyedges=NULL;
	int normalize=TRUE, norm_bin_width=TRUE, validate = TRUE, minzero = FALSE;
	unsigned int ncores = 4;
	// declaration of arguments for processing
	int err = FALSE;
	// potential components of outpus
	uint64_t *corrI = NULL;
	cc_weights **corrl = NULL;
	cc_weights *norm = NULL;
	int out_ndim;
	npy_intp *out_dims = NULL;
	PyObject *out = NULL; // note that this is what will be returned, so that error cleanup doesn't need to be separate from standard cleanup
	char *kwlist[] = {"timesT", "timesU", "bins", "weightsT", "weightsU", "nanosT", "nanosU", "edges", "normalize", "norm_bin_width", "minzero", "validate", "ncores", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|OOOOOppppI", kwlist, &pytimesT, &pytimesU, &pybins, &pyweightsT, &pyweightsU, &pynanosT, &pynanosU, 
										&pyedges, &normalize, &norm_bin_width, &minzero, &validate, &ncores))
		goto exit;
	exit:
	return out;
}*/

static PyMethodDef pyFCS_funcs[] = {
	{"testsequence", (PyCFunction)pyFCS_testsequence, METH_VARARGS|METH_KEYWORDS, "A basic docstring"},
	/*{"correlate", (PyCFunction)pyFCS_correlate, METH_VARARGS|METH_KEYWORDS, 
		"correlate(timesT, timesU, bins, weightsT=None, weightsU=None, nanosT=None, nanosU=None, edges=None, normalize=True, norm_bin_width=True, minzero=False, validate=True, ncores=4)\n"
		"--\n\n"
		"Correlate according all the things that we need\n"
		"Parameters\n"
		"----------\n"
		"timesT/timesU: list[numpy.ndarray]|numpy.ndarray\n"
		"    Arrival times of photons to cross-correlate.\n"
		"    Must be integer array\n"
		"bins: numpy.ndarray\n"
		"weightsT/U: list[numpy.ndarray]|numpy.ndarray, optional\n"
		"    For FLCS, either the weight values for each photon (if\n"
		"    nanosT/U are not specified), or the weights for each\n"
		"    TCSPC channel, channel of each photn specified in nanosT/U.\n"
		"    A 2-D array can be used to specify multiple weights for each\n"
		"    species, in which case the 0th dimension specifies the\n"
		"    species, and the second the TCSPC bin. Should be floating\n"
		"    point array.\n"
		"nanosT/U: list[numpy.ndarray]|numpy.ndarray, optional\n"
		"    The TCSPC time of each photon, must be positive integer array.\n"
		"edges: numpy.ndarray\n"
		"    1 or 2D array, deffining the start/stop times of each burst.\n"
		"    The last dimension must have size = 2.\n"
		"minzero: bool, optional\n"
		"    When using automatic assement of start/stop times for normalization\n"
		"    True will make the start time always 0, False (default),\n"
		"    then use the first photon in the times array(s) for normalization\n"
		"validate: bool, optional\n"
		"    Whether or not to performa a check that times arrays are\n"
		"    monotonically increasing, if True (default), will raise an\n"
		"    error if photons are not monotinically increasing. If false\n"
		"    no error will be raised, but if photons are no monotonically\n"
		"    increasing, the results will be non-defined. Default is True.\n"
		"num_cores: int, optional\n"
		"    Number of cores to use in computation, for parallel processing\n"
		"    optimization (only applicable for multiple bursts).\n"
		"    The default is 4.\n"
		"Returns\n"
		"-------\n"
		"corrl: numpy.ndarray\n"
		"    Correlation array, representing the coorelation\n"
		"    between the T/U arrays. If multiple weights are supplied\n"
		"    first dimension will represent each weight.\n"
		"    last dimension of output will have size one less than the\n"
		"    size of bins.\n"
	},*/
	{NULL,NULL,0,NULL}
};

static struct PyModuleDef pyFCS_module =
{
	PyModuleDef_HEAD_INIT, "pyFCS", 
	"Module for performing binned auto and cross correlations of photon arrival times\n", -1,
	pyFCS_funcs
};	

PyMODINIT_FUNC PyInit_pyFCS(void)
{
	import_array();
	return PyModule_Create(&pyFCS_module);
};
