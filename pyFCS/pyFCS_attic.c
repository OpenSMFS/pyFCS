#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#include "correlate.h"

/*static inline const char* inverse_type(int npy_type)
{
	switch (npy_type)
	{
		case NPY_DOUBLE: return "double";
		case NPY_UINT64: return "uint64";
		case NPY_INT64: return "int64";
		case NPY_INT32: return "int32";
		case NPY_UINT32: return "uint32";
		default: return "unknown";
	}
	return "Unknown";
}*/


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~ Section for NULL-safe frees
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// free with checking that pointer is not NULL
static inline void Xfree(void* arr)
{
	if (arr != NULL)
		free(arr);
}

// free times/weights/nanos pointer arrays
static inline void free_mweights(Py_ssize_t len, double ***ptr)
{
	if (ptr != NULL)
	{
		for (Py_ssize_t i = 0; i < len; i++)
			Xfree(ptr[i]);
		free(ptr);
	}
}

// basically free pointer list, but for 1 level less indirection
static inline void free_edges(Py_ssize_t len, uint64_t **edges)
{
	if (edges != NULL)
	{
		for (Py_ssize_t i = 0; i < len; i++)
			Xfree(edges[i]);
		free(edges);
	}
}

// like Py_XDECREF, used to free child arrays and array of numpy arrays while first checking they exist
static inline void free_nparr(Py_ssize_t len, PyArrayObject **nparrs)
{
	if (nparrs != NULL)
	{
		for(Py_ssize_t i = 0; i < len; i++)
			Py_XDECREF(nparrs[i]);
		free(nparrs);
	}
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~ Section for maping of type num to different strings
// ~ type = 0 -> weights (double) array
// ~ type = 1 -> corrl (double) array
// ~ type = 2 -> times (uint64_t) array
// ~ type = 3 -> nanos (uint16/64_t) array
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// mapping of type num to name for error messages
static inline const char* type_name(int type)
{
	switch (type)
	{
		case 0: return "weigths";
		case 1: return "corrl";
		case 2: return "times";
		case 3: return "nanos";
	}
	return "error in C code";
}

// mapping of type num to data type of array for error messages
static inline const char* type_type(int type)
{
	switch (type)
	{
		case 0: return "floats";
		case 1: return "floats";
		case 2: return "ints";
		case 3: return "ints";
	}
	return "error in C code";
}

// mapping of type num to necessary number of dimensions, for error messages
static inline const char* type_ndstr(int type)
{
	switch (type)
	{
		case 0: return "1/2";
		case 1: return "1/2";
		case 2: return "1";
		case 3: return "1";
	}
	return "error in C code";
}

// mapping of type num to the minimum last dimension's size
static inline npy_intp type_minlen(int type)
{
	switch (type)
	{
		case 0: return 1;
		case 1: return 1;
		case 2: return 3;
		case 3: return 3;
	}
	return -1;
}

// mapping of type num to numpy type code
static inline int type_typenum(int type)
{
	switch (type)
	{
		case 0: return NPY_DOUBLE;
		case 1: return NPY_DOUBLE;
		case 2: return NPY_UINT64;
		case 3: return NPY_UINT64;
		
	}
	return 0;
}

// mapping of type num to min NDIM  of arrays
static inline int type_maxNdim(int type)
{
	switch (type)
	{
		case 0: return 2;
		case 1: return 2;
		case 2: return 1;
		case 3: return 1;
		
	}
	return 1;
}

// generate the name of array (usually with error)
static inline char* type_namestr(int type, char *TU, Py_ssize_t pos)
{
	static char outstr[20];
	if (pos == -1)
		sprintf(outstr, "%s%s", type_name(type), TU);
	else
		sprintf(outstr, "%s%s[%ld]", type_name(type), TU, pos);
	return outstr;
}

// check that arrays have propper dimensionallity of 1st dimentions
static inline int bool_ndim(PyArrayObject *arr, int type)
{
	int nd = PyArray_NDIM(arr);
	if (nd < 1) 
		return TRUE;
	else if (nd > type_maxNdim(type)) 
		return TRUE;
	else if ( PyArray_DIM(arr, nd-1) < type_minlen(type))
		return TRUE;
	return FALSE;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~ Checking functions, these verify some characteristic, and have no
// ~ other output.
// ~ Returns TRUE if check **fails** (so that ! is not needed in if
// ~ statements
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ensure last dimensions of paired lists of arrays are the same
static inline int check_lengths(Py_ssize_t len, size_t *lensT, size_t *lensU, PyArrayObject **arrsT, PyArrayObject **arrsU, int type)
{
	int check_dim = PyArray_NDIM(arrsT[0]) - 1;
	for (Py_ssize_t i; i < len; i++)
	{
		if (lensT[i] != PyArray_DIM(arrsT[i], check_dim))
		{
			PyErr_Format(PyExc_ValueError, "mismatched lenghts of timesT[%lld] and %sT[%lld] arrays, got %llu and %lld", i, type_name(type), i, lensT[i], PyArray_DIM(arrsT[i], check_dim));
			return TRUE;
		}
		if (lensU[i] != PyArray_DIM(arrsU[i], check_dim))
		{
			PyErr_Format(PyExc_ValueError, "mismatched lenghts of timesU[%lld] and %sU[%lld] arrays, got %llu and %lld", i, type_name(type), i, lensU[i], PyArray_DIM(arrsU[i], check_dim));
			return TRUE;
		}
	}
	return FALSE;
}

// check times arrays are monotonically increasing
static inline int check_monotonicincrease(Py_ssize_t len, size_t *lenT, size_t *lenU, uint64_t **timesT, uint64_t **timesU)
{
	Py_ssize_t i;
	size_t j, jj;
	for (i = 0; i < len; i++)
	{
		for (j = 0, jj = 1; jj < lenT[i]; j++, jj++)
		{
			if (timesT[i][j] > timesT[i][jj])
			{
				PyErr_Format(PyExc_ValueError, "timesT[%lld] array is not monotonically increasing", i);
				return TRUE;
			}
		}
		for (j = 0, jj = 1; jj < lenU[i]; j++, jj++)
		{
			if (timesU[i][j] > timesU[i][jj])
			{
				PyErr_Format(PyExc_ValueError, "timesU[%lld] array is not monotonically increasing", i);
				return TRUE;
			}
		}
	}
	return FALSE;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~ verify functions similar to check, but have more varied output.
// ~ Returns negative numbers for fail, while 0 and possitive numbers
// ~ indicate what type of process indicated by verification.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// verify both inputs are null (0), both filled (1) or mismatched (-1)
static inline int verify_allornone(PyObject *pyarrsA, PyObject *pyarrsB)
{
	if ((pyarrsA != NULL)!=(pyarrsB != NULL))
		return -1;
	if (pyarrsA == NULL)
		return 0;
	return 1;
}

// Check valid combination of kwargs specified
// (0) for no weigths or nanos, (1) for only weights, (2) weights+nanos
static inline int verify_kwargcombs(PyObject* weightsT, PyObject* weightsU, PyObject* nanosT, PyObject* nanosU)
{
	int w = verify_allornone(weightsT, weightsU);
	if (w == -1)
	{
		PyErr_SetString(PyExc_ValueError, "Values for weigthsT/U must both be specified or neither");
		return -1;
	}
	int n = verify_allornone(nanosT, nanosU);
	if (n == -1)
	{
		PyErr_SetString(PyExc_ValueError, "Values for nanosT/U must both be specified or neither");
		return -1;
	}
	if ((w == 0)&(n == 0))
		return 0;
	else if ((w==1)&(n == 0))
		return 1;
	else if ((w==1)&(n == 1))
		return 2;
	PyErr_SetString(PyExc_ValueError, "nanosT/U cannot be specified without also specifiying weightsT/U");
	return -1;
}

// interpret the normalize keyword argument
static inline int verify_normalize(PyObject *pynormalize)
{
	if (pynormalize == NULL)
		return 2;
	if (PyUnicode_Check(pynormalize))
	{
		if (PyUnicode_CompareWithASCIIString(pynormalize, "HIST")|PyUnicode_CompareWithASCIIString(pynormalize, "hist"))
			return 0;
		else
		{
			PyErr_SetString(PyExc_ValueError, "norm must be True, False, or 'hist'");
			return -1;
		}
	}
	if (PyObject_Not(pynormalize) > 0)
		return 1;
	if (PyObject_IsTrue(pynormalize) > 0)
		return 2;
	PyErr_SetString(PyExc_ValueError, "norm must be True, False, or 'hist'");
	return -1;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~ Cast functions for converting array to numpy arrays, returns array
// ~ NULL on fail
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// function to cast to array to appropriate type (after separated arrays in input)
static inline PyArrayObject* cast_array(PyObject *arr, int type, char *TU, Py_ssize_t pos)
{
	PyArrayObject *out = NULL;
	if (type < 2)
	{
		out = (PyArrayObject*) PyArray_FROM_OTF(arr, type_typenum(type), NPY_ARRAY_IN_ARRAY);
	}
	else
	{
		PyArrayObject *temp = (PyArrayObject*) PyArray_FROM_O(arr);
		if ((temp != NULL)&(PyArray_ISINTEGER(temp)))
		{
			out = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) temp, type_typenum(type), NPY_ARRAY_IN_ARRAY|NPY_FORCECAST);
		}
		Py_XDECREF(temp);
	}
	if (out == NULL)
	{
		PyErr_Format(PyExc_TypeError, "%s must be %sD array of %s with greater than %d elements", type_namestr(type, TU, pos), type_ndstr(type), type_type(type), type_minlen(type));
		return NULL;
	}
	if (bool_ndim(out, type))
	{
		Py_DECREF(out);
		PyErr_Format(PyExc_TypeError, "%s must be %sD array of %s with greater than %d elements", type_namestr(type, TU, pos), type_ndstr(type), type_type(type), type_minlen(type));
		return NULL;
	}
	return out;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~ Get functions retrieve particular information from processed numpy
// ~ arrays, these return their output, and do not take pointers to
// ~ write to
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// get the lengths of each array in list of numpy arrays
static inline size_t* get_sizes(Py_ssize_t len, PyArrayObject **nparrs)
{
	size_t *lens;
	if ((lens = (size_t*) calloc(len, sizeof(size_t))) == NULL)
	{
		PyErr_NoMemory();
		return NULL;
	}
	for (Py_ssize_t i = 0; i < len; i++)
		lens[i] = (size_t) PyArray_DIM(nparrs[i], 0);
	return lens;
}

// extract the data pointers from array of numpy arrays
static inline void** get_datas(Py_ssize_t len, size_t ptrsz, PyArrayObject **nparr)
{
	void **out = (void**) calloc(len, ptrsz);
	if (out == NULL)
	{
		PyErr_NoMemory();
		return NULL;
	}
	for (Py_ssize_t i = 0; i < len; i++)
		out[i] = PyArray_DATA(nparr[i]);
	return out;
}

// extract base set of edges specifically from times, allocates arrays as well
static inline uint64_t** get_edgesfromtimes(int minzero, Py_ssize_t len, size_t *lenT, size_t *lenU, uint64_t **timesT, uint64_t **timesU)
{
	uint64_t **edges = NULL;
	Py_ssize_t i;
	if ((edges = (uint64_t**) calloc(len, sizeof(uint64_t*)))==NULL)
		goto err;
	for (i = 0; i < len; i++)
	{
		if ((edges[i] = (uint64_t*) malloc(2*sizeof(uint64_t)))==NULL)
			goto err;
		edges[i][1] = timesT[i][lenT[i]-1] > timesU[i][lenU[i]-1] ? timesT[i][lenT[i]-1] : timesU[i][lenU[i]-1];
	}
	if (minzero)
	{
		for (i = 0; i < len; i++)
			edges[i][0] = 0;
	}
	else
	{
		for (i = 0; i < len; i++)
			edges[i][0] = timesT[i][0] < timesU[i][0] ? timesT[i][0] : timesU[i][0];	
	}
	for (i = 0; i < len; i++)
		
	return edges;
	err:
	PyErr_NoMemory();
	free_edges(len, edges);
	return NULL;
}

// verify edges specified in kwargs are valid and re-place values into edges array
// this function is names set_ instead of get because it does modify edges
// but it does not allocate edges (it assumes these are already passed through
// get_edgesfromtimes)
static inline int set_edgesfrompy(Py_ssize_t len, PyObject *pyedges, uint64_t **edges)
{
	Py_ssize_t i, n;
	uint64_t *edges_data;
	PyArrayObject *npedges = (PyArrayObject*) PyArray_FROMANY(pyedges, NPY_UINT64, 1, 2, NPY_ARRAY_IN_ARRAY|NPY_FORCECAST);
	if (npedges == NULL)
		goto nperr;
	int ndim = PyArray_NDIM(npedges);
	if ((ndim > 2)|(ndim < 1)) // from any should stop this from creating an error
		goto nperr;
	if (PyArray_DIM(npedges, ndim - 1) != 2) // last dimension should be 2 ie represent [start, stop] of each burst
		goto nperr;
	if ((ndim == 2) & (PyArray_DIM(npedges, 0) != len)) // if 2D, must define [start, stop] of each burst, 1D, assume all the same
		goto nperr;
	edges_data = (uint64_t*) PyArray_DATA(npedges);
	for ( i = 0 ; i < len ; i++ )
	{
		n = (ndim == 2) ? 2*i : 0; // when all 
		if (edges[i][0] < edges_data[n])
			goto vaerr;
		if (edges[i][1] > edges_data[n+1])
			goto vaerr;
		edges[i][0] = edges_data[n];
		edges[i][1] = edges_data[n+1];
	}
	Py_DECREF(npedges);
	return FALSE;
	vaerr:
	PyErr_SetString(PyExc_ValueError, "edges values have min/max greater/less than values start/stop times of arrays");
	Py_XDECREF(npedges);
	return TRUE;
	nperr:
	PyErr_SetString(PyExc_ValueError, "edges must be array with last dimension of 2 and 0th identical to number of pairs of arrays");
	Py_XDECREF(npedges);
	return TRUE;
}

// process the arguments for edges (start/stop) of times
static inline uint64_t** get_edges(int minzero, Py_ssize_t len, size_t *lenT, size_t *lenU, uint64_t **timesT, uint64_t **timesU, PyObject *pyedges)
{
	uint64_t **edges = NULL;
	if ((pyedges != NULL) & !Py_IsNone(pyedges)) // if pyedges exists, the ignore minzero
	{
		if (minzero)
			PyErr_WarnEx(PyExc_UserWarning, "when edges are specified, minzero is ignored/treated as False", 1);
		minzero = FALSE;
	}
	if ((edges = get_edgesfromtimes(minzero, len, lenT, lenU, timesT, timesU)) == NULL)
		return NULL;
	if ((pyedges != NULL) & !Py_IsNone(pyedges))
	{
		if (set_edgesfrompy(len, pyedges, edges))
		{
			free_edges(len, edges);
			return NULL;
		}
	}
	return edges;
}

// check weigths arrays are all matching number of 0th dim
static inline size_t get_weights_dim(Py_ssize_t len, PyArrayObject **npweightsT, PyArrayObject **npweightsU)
{
	Py_ssize_t i;
	int ndim = PyArray_NDIM(npweightsT[0]);
	if ((ndim != 1) & (ndim != 2))
	{
		PyErr_SetString(PyExc_ValueError, "weightsT/U cannot be more than 2D arrays");
		return 0;
	}
	npy_intp nW = ndim == 1 ? 1 : PyArray_DIM(npweightsT[0], 0);
	if (nW < 1)
	{
		PyErr_SetString(PyExc_ValueError, "weigthsT/U cannot be 0 size array");
		return 0;
	}
	if (ndim == 1)
	{
		for (i = 0; i < len; i++)
		{
			if ((PyArray_NDIM(npweightsT[i]) != 1)|(PyArray_NDIM(npweightsU[i]) != 1))
				return 0;
		}
	}
	else
	{
		for (i = 0; i < len; i++)
		{
			if ((PyArray_NDIM(npweightsT[i])!=2)|(PyArray_NDIM(npweightsU[i])!=2)|(PyArray_DIM(npweightsT[i],0) != nW)|(PyArray_DIM(npweightsU[i],0) != nW))
			{
				PyErr_SetString(PyExc_ValueError, "weightsT/U arrays have inconsisent number of filter functions (0th dimension)");
				return 0;
			}
		}
	}
	return (size_t) nW;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~ parse functions used to return some Py_ssize_t indicating the size
// ~ and use pointers as input to return pointer arrays to PyArrayObjects
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// parse an input arrays, from T or U, used in function to parse together
static inline Py_ssize_t parse_single(PyObject *pyarrs, int type, char *TU, PyArrayObject ***nparrs)
{
	Py_ssize_t len, i , j;
	PyObject *iter = NULL, *next=NULL;
	PyArrayObject *temp = NULL;
	PyArrayObject **oarrs = NULL;
	PyObject *pynpcast = PyArray_FROM_O(pyarrs);
	if (pynpcast == NULL)
	{
		PyErr_Format(PyExc_TypeError, "Cannot interpret %s%s: must be %sD array of %s or sequence thereof", type_name(type), TU, type_ndstr(type), type_type(type));
		return 0;
	}
	// case where input is array, and dtype is not object (of object assume ragged array)
	if (PyArray_Check(pynpcast) & !PyArray_ISOBJECT((PyArrayObject*) pynpcast))
	{
		if ((temp = cast_array(pynpcast, type, TU, -1)) == NULL)
			return 0;
		if((oarrs = (PyArrayObject**) malloc(sizeof(PyArrayObject*))) == NULL)
		{
			PyErr_NoMemory();
			Py_DECREF(temp);
			Py_DECREF(pynpcast);
			return 0;
		}
		oarrs[0] = temp;
		len = 1;
	}
	else
	{
		len = PyObject_Length(pynpcast);
		if ( len < 1) // catches both case of 0 length list or non-iterable object
		{
			PyErr_Format(PyExc_TypeError, "Cannot interpret %s%s: must be %sD array of %s or sequence thereof", type_name(type), TU, type_ndstr(type), type_type(type));
			Py_DECREF(pynpcast);
			return -1;
		}
		if ((iter = PyObject_GetIter(pynpcast)) == NULL)
		{
			PyErr_BadInternalCall();
			return 0;
		}
		if ((oarrs = (PyArrayObject**) calloc(len, sizeof(PyArrayObject*))) == NULL)
		{
			PyErr_NoMemory();
			Py_DECREF(iter);
			Py_DECREF(pynpcast);
			return 0;
		}
		for ( i = 0; i < len; i++)
		{
			if ((next = PyIter_Next(iter))==NULL)
			{
				for ( j = 0; j < i; j++)
					Py_DECREF(oarrs[j]);
				Py_DECREF(iter);
				Py_DECREF(pynpcast);
				return 0;
			}
			if((oarrs[i] = cast_array(next, type, TU, i))==NULL)
			{
				for ( j = 0; j < i; j++)
					Py_DECREF(oarrs[j]);
				Py_DECREF(iter);
				Py_DECREF(next);
				Py_DECREF(pynpcast);
				return 0;
			}
			Py_DECREF(next);
		}
		Py_DECREF(iter);
	}
	Py_DECREF(pynpcast);
	*nparrs = oarrs;
	return len;
}

// for T/U arrays, extract together into cast numpy arrays
static inline Py_ssize_t parse_input(PyObject *pyarrT, PyObject *pyarrU, int type, Py_ssize_t expected_length, PyArrayObject ***nparrT, PyArrayObject ***nparrU)
{
	Py_ssize_t lenT = 0, lenU = 0;
	PyArrayObject **oarrT = NULL, **oarrU = NULL;
	lenT = parse_single(pyarrT, type, "T", &oarrT);
	if (lenT < 1)
		goto fail;
	lenU = parse_single(pyarrU, type, "U", &oarrU);
	if (lenU < 1)
		goto fail;
	if (lenT != lenU)
	{
		PyErr_Format(PyExc_ValueError, "Mismatched number of arrays in %sT/U, got %ld and %ld respectively", type_name(type), lenT, lenU);
		goto fail;
	}
	if ((expected_length > 0)&(lenT != expected_length))
	{
		PyErr_Format(PyExc_ValueError, "Mismatched number of arrays in %sT/U and timesT/U, got %lld and %lld respectively", type_name(type), lenT, expected_length);
		goto fail;
	}
	*nparrT = oarrT;
	*nparrU = oarrU;
	return lenT;
	fail:
	free_nparr(lenT, oarrT);
	free_nparr(lenU, oarrU);
	return 0;
}

// for weights without nanos, get the number of weights arrays and allocate data to weightsTT/weightsUU
static inline size_t parse_weights(Py_ssize_t lentimes, size_t *lenT, size_t *lenU, PyArrayObject **npweightsT, PyArrayObject **npweightsU, 
								double ****weightsTT, double ****weightsUU)
{
	size_t nweight = 0;
	double **weightsT = NULL, **weightsU=NULL;
	double ***oweightsTT=NULL, ***oweightsUU=NULL;
	if (check_lengths(lentimes, lenT, lenU, npweightsT, npweightsU, 0))
		return 0;
	if ((nweight = get_weights_dim(lentimes, npweightsT, npweightsU))==0)
		return 0;
	if ((weightsT = (double**) get_datas(lentimes, sizeof(double*), npweightsT))==NULL)
		goto err;
	if ((weightsU = (double**) get_datas(lentimes, sizeof(double*), npweightsU))==NULL)
		goto err;
	if ((oweightsTT = malloc(lentimes*sizeof(double**)))==NULL)
		goto memerr;
	if ((oweightsUU = malloc(lentimes*sizeof(double**)))==NULL)
		goto memerr;
	// distribute between arrays
	for (Py_ssize_t i = 0; i < lentimes; i++)
	{
		if ((oweightsTT[i] = malloc(nweight*sizeof(double**)))==NULL)
			goto memerr;
		if ((oweightsUU[i] = malloc(nweight*sizeof(double**)))==NULL)
			goto memerr;
		for (size_t j = 0; j < nweight; j++)
		{
			oweightsTT[i][j] = weightsT[i]+(j*lenT[i]);
			oweightsUU[i][j] = weightsU[i]+(j*lenU[i]);
		}
	}
	Xfree(weightsT);
	Xfree(weightsU);
	*weightsTT = oweightsTT;
	*weightsUU = oweightsUU;
	return nweight;
	memerr:
	PyErr_NoMemory();
	err:
	Xfree(weightsT);
	Xfree(weightsU);
	free_mweights(lentimes, oweightsTT);
	free_mweights(lentimes, oweightsUU);
	return 0;
}

// extract weights and nanos from arrays with error checking
static inline size_t parse_weights_nanos(Py_ssize_t len, size_t *lenT, size_t *lenU, 
										PyArrayObject **npweightsT, PyArrayObject **npweightsU,
										PyArrayObject **npnanosT, PyArrayObject **npnanosU,
										double ***weightsT, double ***weightsU, uint64_t ***nanosT, uint64_t ***nanosU)
{
	size_t nweight;
	uint64_t **onanosT = NULL, **onanosU = NULL;
	npy_intp wdT, wdU;
	double **oweightsT = NULL, **oweightsU = NULL;
	if ((nweight = get_weights_dim(1, npweightsT, npweightsU)) == 0)
		return 0;
	if (check_lengths(len, lenT, lenU, npnanosT, npnanosU, 3))
		return 0;
	if ((onanosT = (uint64_t**) get_datas(len, sizeof(uint64_t*), npnanosT))==NULL)
		return 0;
	if ((onanosU = (uint64_t**) get_datas(len, sizeof(uint64_t*), npnanosU))==NULL)
	{
		PyErr_NoMemory();
		goto err;
	}
	// get max index in nanos, for checking size of weights
	uint64_t maxT=0, maxU=0;
	Py_ssize_t i;
	size_t j;
	for (i = 0; i < len; i++)
	{
		for (j = 0; j < lenT[i]; j++)
		{
			if (onanosT[i][j] > maxT)
				maxT = onanosT[i][j];
		}
		for (j = 0; j < lenU[i]; j++)
		{
			if (onanosU[i][j] > maxU)
				maxU = onanosU[i][j];
		}
	}
	// check size of weights arrays
	if ((wdT = PyArray_DIM(npweightsT[0], PyArray_NDIM(npweightsT[0])-1)) <= maxT)
	{
		PyErr_Format(PyExc_ValueError, "weightsT insufficient indeces for specified nanosT array (has %ld, but requires at least %ld)", wdT, maxT);
		goto err;
	}
	if ((wdU = PyArray_DIM(npweightsU[0], PyArray_NDIM(npweightsU[0])-1)) <= maxU)
	{
		PyErr_Format(PyExc_ValueError, "weightsU insufficient indeces for specified nanosT array (has %ld, but requires at least %ld", wdU, maxU);
		goto err;
	}
	// get datas arrays
	if ((oweightsT = (double**) malloc(nweight*sizeof(double*)))==NULL)
		goto memerr;
	if ((oweightsU = (double**) malloc(nweight*sizeof(double*)))==NULL)
		goto memerr;
	if ((oweightsT = (double**) malloc(nweight*sizeof(double*)))==NULL)
		goto memerr;
	if ((oweightsU = (double**) malloc(nweight*sizeof(double*)))==NULL)
		goto memerr;
	double* tempweightsT = (double*) PyArray_DATA(npweightsT[0]);
	double* tempweightsU = (double*) PyArray_DATA(npweightsU[0]);
	for (size_t ii = 0; ii < nweight; ii++)
	{
		oweightsT[ii] = tempweightsT+(wdT*ii);
		oweightsU[ii] = tempweightsU+(wdU*ii);
	}
	*nanosT = onanosT;
	*nanosU = onanosU;
	*weightsT = oweightsT;
	*weightsU = oweightsU;
	return nweight;
	memerr:
	PyErr_NoMemory();
	err:
	Xfree(oweightsT);
	Xfree(oweightsU);
	Xfree(onanosT);
	Xfree(onanosU);
	return 0;
}

// get bins of array
static inline size_t parse_bins(PyObject *pybins, PyArrayObject **npbins, uint64_t **bins)
{
	size_t nbin = 0;
	PyArrayObject *obins = (PyArrayObject*) PyArray_FROM_OTF(pybins, NPY_UINT64, NPY_ARRAY_IN_ARRAY|NPY_FORCECAST);
	if (obins == NULL)
		return 0;
	if (PyArray_NDIM(obins) != 1)
	{
		PyErr_SetString(PyExc_ValueError, "bins must be 1D numpy array");
		return 0;
	}
	if ((nbin = (size_t) PyArray_DIM(obins, 0)) < 2)
	{
		PyErr_SetString(PyExc_ValueError, "bins must have at least 2 values");
		return 0;
	}
	*npbins = obins;
	*bins = PyArray_DATA(obins);
	return nbin;
}

static inline int allocate_corrI(size_t nbins, uint64_t **corrI, int *outndim, npy_intp **outdim)
{
	if ((*corrI = calloc(nbins, sizeof(uint64_t))) == NULL)
	{
		PyErr_NoMemory();
		return TRUE;
	}
	if ((*outdim = malloc(sizeof(npy_intp))) == NULL)
	{
		PyErr_NoMemory();
		free(*corrI);
		*corrI = NULL;
		return TRUE;
	}
	**outdim = (npy_intp) nbins;
	*outndim = 1;
	return FALSE;
}

static inline int allocate_corrl(size_t nbins, size_t nweights, int ndim, double **corrl, int *outndim, npy_intp **outdim)
{
	if ((*corrl = calloc(nbins*nweights, sizeof(double))) == NULL)
	{
		PyErr_NoMemory();
		return TRUE;
	}
	if ((*outdim = malloc(ndim*sizeof(npy_intp))) == NULL)
	{
		free(*corrl);
		*corrl = NULL;
		return TRUE;
	}
	switch (ndim)
	{
		case 1:
			**outdim = (npy_intp) nbins;
			break;
		case 2:
			outdim[0][0] = (npy_intp) nweights;
			outdim[0][1] = (npy_intp) nbins;
			break;
		default:
			free(*corrl);
			*corrl = NULL;
			free(*outdim);
			*outdim = NULL;
			PyErr_BadInternalCall();
			return TRUE;
	}
	*outndim = (int) ndim;
	return FALSE;
}

// Python correlate function
static PyObject *pyFCS_correlate(PyObject *self, PyObject *args, PyObject *kwargs)
{
	// inital parsing of arguments
	char *kwlist[] = {"timesT", "timesU", "bins", "weightsT", "weightsU", "nanosT", "nanosU", "edges", "normalize", "minzero", "validate", "ncores", NULL};
	PyObject *pytimesT=NULL, *pytimesU=NULL, *pybins=NULL, *pyweightsT=NULL, *pyweightsU=NULL, *pynanosT=NULL, *pynanosU=NULL, *pyedges=NULL, *pynormalize=NULL;
	PyObject *out = NULL; // note that this is what will be returned, so that error cleanup doesn't need to be separate from standard cleanup
	PyArrayObject **nptimesT=NULL, **nptimesU=NULL, **npweightsT=NULL, **npweightsU=NULL, **npnanosT=NULL, **npnanosU=NULL, *npbins=NULL;
	Py_ssize_t lentimes=0, lenweights=0, lennanos=0;
	int validate = TRUE, minzero = FALSE, outndim = 0, func_code = -1, norm_code = -1;
	unsigned int ncores = 4;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|OOOOOOppI", kwlist, &pytimesT, &pytimesU, &pybins, &pyweightsT, &pyweightsU, &pynanosT, &pynanosU, &pyedges, &pynormalize, &minzero, &validate, &ncores))
		return NULL;
	size_t nweight = 0, nbins = 0, lenbins=0, i, j;
	size_t *lenT=NULL, *lenU=NULL;
	uint64_t **timesT=NULL, **timesU=NULL, **nanosT=NULL, **nanosU=NULL, **edges=NULL, *bins=NULL, *corrI=NULL;
	double **weightsT=NULL, **weightsU=NULL, ***weightsTT=NULL, ***weightsUU=NULL, *norm=NULL, *corrl=NULL;
	npy_intp *outdim = NULL;
	// parse the normalize argument
	if ((norm_code = verify_normalize(pynormalize)) == -1)
		return NULL;
	// checking that weights/nanosT/U are defined together
	if ((func_code = verify_kwargcombs(pyweightsT, pyweightsU, pynanosT, pynanosU)) == -1)
		return NULL;
	// get sizes and convert to numpy array(s)
	// select which weights/nanos to extract based on proper combination
	if ((lenbins = parse_bins(pybins, &npbins, &bins))==0)
		return NULL;
	nbins = lenbins - 1;
	if ((lentimes = parse_input(pytimesT, pytimesU, 2, 0, &nptimesT, &nptimesU)) == 0)
		goto memfree;
	if ((lenT = get_sizes(lentimes, nptimesT)) == NULL)
		goto memfree;
	if ((lenU = get_sizes(lentimes, nptimesU)) == NULL)
		goto memfree;
	if ((timesT = (uint64_t**) get_datas(lentimes, sizeof(uint64_t*), nptimesT))==NULL)
		goto memfree;
	if ((timesU = (uint64_t**) get_datas(lentimes, sizeof(uint64_t*), nptimesU))==NULL)
		goto memfree;
	if ((validate)& (check_monotonicincrease(lentimes, lenT, lenU, timesT, timesU)))
		goto memfree;
	if ((edges = get_edges(minzero, lentimes, lenT, lenU, timesT, timesU, pyedges)) == NULL)
		goto memfree;
	// allocate the correlation output arrays
	switch (func_code)
	{
		case 0:
			if (allocate_corrI(nbins, &corrI, &outndim, &outdim))
				goto memfree;
			break;
		case 1:
			// process the weights w/o nanos
			if ((lenweights = parse_input(pyweightsT, pyweightsU, 0, lentimes, &npweightsT, &npweightsU)) == 0)
				goto memfree;
			if ((nweight = parse_weights(lenweights, lenT, lenU, npweightsT, npweightsU, &weightsTT, &weightsUU)) == 0)
				goto memfree;
			if (allocate_corrl(nbins, nweight, PyArray_NDIM(npweightsT[0]), &corrl, &outndim, &outdim))
				goto memfree;
			break;
		case 2:
			// process weights w/ nanos
			if ((lenweights = parse_input(pyweightsT, pyweightsU, 0, 1, &npweightsT, &npweightsU)) != 1)
			{
				PyErr_SetString(PyExc_ValueError, "when nanosT/U are specified, weightsT/U must be 1/2D array with last dimension larger than largest index in nanosT/U");
				goto memfree;
			}
			// process nanos
			if ((lennanos = parse_input(pynanosT, pynanosU, 3, lentimes, &npnanosT, &npnanosU)) != lentimes)
				goto memfree;
			if ((nweight = parse_weights_nanos(lennanos, lenT, lenU, npweightsT, npweightsU, npnanosT, npnanosU, &weightsT, &weightsU, &nanosT, &nanosU)) == 0)
				goto memfree;
			if (allocate_corrl(nbins, nweight, PyArray_NDIM(npweightsT[0]), &corrl, &outndim, &outdim))
				goto memfree;
			break;
	}
	// if needed allocate the normalization array
	if (norm_code ==  2)
	{
		if ((norm = calloc(nbins, sizeof(double*))) == NULL)
		{
			PyErr_NoMemory();
			goto memfree;
		}
	}
	int err = -1;
	// perform the appropriate correlation function
	switch (func_code)
	{
		case 0:
			err = correlate_parallel(lentimes, edges, lenT, timesT, lenU, timesU, lenbins, bins, corrI, norm, ncores);
			break;
		case 1:
			err = correlate_weight_parallel(lentimes, edges, nweight, lenT, timesT, weightsTT, lenU, timesU, weightsUU, lenbins, bins, corrl, norm, ncores);
			break;
		case 2:
			err = correlate_weight_index_parallel(lentimes, edges, nweight, lenT, timesT, weightsT, nanosT, lenU, timesU, weightsU, nanosU, lenbins, bins, corrl, norm, ncores);
			break;
	}
	if (err)
		goto memfree;
	if (func_code == 0)
	{
		switch (norm_code)
		{
			case 0:
				out = PyArray_SimpleNewFromData(outndim, outdim, NPY_UINT64, corrI);
				break;
			case 1:
				if ((corrl = calloc(nbins, sizeof(double)))==NULL)
				{
					PyErr_NoMemory();
					goto memfree;
				}
				bin_norm(lenbins, bins, corrI, corrl);
				out = PyArray_SimpleNewFromData(outndim, outdim, NPY_DOUBLE, corrl);
				free(corrI);
				break;
			case 2:
				if ((corrl = calloc(nbins, sizeof(double)))==NULL)
				{
					PyErr_NoMemory();
					goto memfree;
				}
				bin_norm(lenbins, bins, corrI, corrl);
				for( i = 0; i < nbins; i++ )
					corrl[i] = corrl[i] * norm[i];
				free(corrI);
				out = PyArray_SimpleNewFromData(outndim, outdim, NPY_DOUBLE, corrl);
				break;
			
		}
	}
	else
	{
		switch (norm_code)
		{
			case 2:
				for (i = 0; i < nweight; i++)
				{
					for (j = 0; j < nbins; j++)
						corrl[(i*nbins)+j] *= norm[j];
				}
			case 1:
				if (bin_norm_multi_w_flat(nweight, lenbins, bins, corrl))
					goto memfree;
			case 0:
				out = PyArray_SimpleNewFromData(outndim, outdim, NPY_DOUBLE, corrl);
		}
	}
	memfree:
	Xfree(outdim);
	Xfree(nanosU);
	Xfree(nanosT);
	free_nparr(lennanos, npnanosU);
	free_nparr(lennanos, npnanosT);
	Xfree(weightsU);
	Xfree(weightsT);
	free_mweights(lenweights, weightsUU);
	free_mweights(lenweights, weightsTT);
	free_nparr(lenweights, npweightsU);
	free_nparr(lenweights, npweightsT);
	Xfree(norm);
	free_edges(lentimes, edges);
	free_nparr(lentimes, nptimesU);
	free_nparr(lentimes, nptimesT);
	Xfree(timesU);
	Xfree(timesT);
	Xfree(lenU);
	Xfree(lenT);
	Py_XDECREF(npbins);
	return out;
}

static inline size_t parse_corrl(PyObject *pycorrl, Py_ssize_t nbins, PyArrayObject **npcorrl, double ***corrl)
{
	double **ocorrl = NULL;
	PyArrayObject* onpcorrl = (PyArrayObject*) PyArray_FROMANY(pycorrl, 1,2, NPY_UINT64, NPY_ARRAY_IN_ARRAY|NPY_FORCECAST|NPY_ARRAY_ENSURECOPY);
	double *npdata;
	if (onpcorrl == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "corrl must be 1 or 2/D array with last dimension of size 1 less than size of bins array");
		return 0;
	}
	if (nbins != PyArray_DIM(onpcorrl, PyArray_NDIM(onpcorrl) - 1))
	{
		Py_DECREF(onpcorrl);
		PyErr_SetString(PyExc_ValueError, "corrl must be 1 or 2/D array with last dimension of size 1 less than size of bins array");
		return 0;
	}
	size_t nW = (PyArray_NDIM(onpcorrl) == 1) ? 1 : (size_t) PyArray_DIM(onpcorrl, 1);
	if ((ocorrl = calloc(nW, sizeof(double**))) == NULL)
	{
		PyErr_NoMemory();
		Py_DECREF(onpcorrl);
		return 0;
	}
	npdata = (double*) PyArray_DATA(onpcorrl);
	for (size_t i = 0; i < nW; i++)
		ocorrl[i] = npdata + (i*nbins);
	*npcorrl = onpcorrl;
	*corrl = ocorrl;
	return nW;
}

static PyObject *pyFCS_normalize(PyObject *self, PyObject *args, PyObject *kwargs)
{
	char *kwlist[] = {"corrl", "bins", "timesT", "timesU", "edges", "norm_bin_width", "minzero", "validate", NULL};
	PyObject *pycorrl = NULL, *pybins=NULL, *pytimesT=NULL, *pytimesU=NULL, *pyedges=NULL;
	int minzero = FALSE, norm_code = -1, norm_bw=TRUE, validate=TRUE;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOppp", kwlist, &pycorrl, &pybins, &pytimesT, &pytimesU, &pyedges, &norm_bw, &minzero, &validate))
		return NULL;
	if ((norm_code = verify_allornone(pytimesT, pytimesU)) == -1)
	{
		PyErr_SetString(PyExc_ValueError, "must specify both timesT/U or neither");
		return NULL;
	}
	PyArrayObject *npbins=NULL, *npcorrl, **nptimesT=NULL, **nptimesU=NULL;
	double **corrl = NULL;
	Py_ssize_t lentimes = 0, nbins=0, nweight = 0;
	size_t *lensT=NULL, *lensU=NULL;
	uint64_t **edges = NULL, **timesT = NULL, **timesU = NULL, *bins=NULL;
	// start extracting/converting data
	if ((nbins = parse_bins(pybins, &npbins, &bins)) < 1)
		goto memfree;
	if ((nweight = parse_corrl(pycorrl, nbins-1, &npcorrl, &corrl))==0)
		goto memfree;
	// extract times and edges
	if (norm_code)
	{
		if ((lentimes = parse_input(pytimesT, pytimesU, 0, 0, &nptimesT, &nptimesU) < 1))
			goto memfree;
		lensT = get_sizes(lentimes, nptimesT);
		lensU = get_sizes(lentimes, nptimesU);
		if ((timesT = (uint64_t**) get_datas(lentimes, sizeof(uint64_t*), nptimesT)) == NULL)
			goto memfree;
		if ((timesU = (uint64_t**) get_datas(lentimes, sizeof(uint64_t*), nptimesU)) == NULL)
			goto memfree;
		if (validate & check_monotonicincrease(lentimes, lensT, lensU, timesT, timesU))
			goto memfree;
		if ((edges = get_edges(minzero, lentimes, lensT, lensU, timesT, timesU, pyedges)) == NULL)
			goto memfree;
		if (normalize_array_multi_multi(nweight, lentimes, nbins, lensT, timesT, lensU, timesT, edges, bins, corrl))
			goto memfree;
	}
	// normalize bins
	if (norm_bw)
		bin_norm_multi_w(nweight, nbins, bins, corrl);
	// generate output
	memfree:
	free_edges(lentimes, edges);
	Xfree(timesT);
	Xfree(timesU);
	Xfree(lensT);
	Xfree(lensU);
	free_nparr(lentimes, nptimesT);
	free_nparr(lentimes, nptimesU);
	Xfree(bins);
	Py_XDECREF(npbins);
	return (PyObject*) npcorrl;
}

static PyMethodDef pyFCS_funcs[] = {
	{"correlate", (PyCFunction)pyFCS_correlate, METH_VARARGS|METH_KEYWORDS, 
		"correlate(timesT, timesU, bins, weightsT=None, weightsU=None, nanosT=None, nanosU=None, edges=None, normalize=True, minzero=False, validate=True, ncores=4)\n"
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
	},
	{"normalize", (PyCFunction)pyFCS_normalize, METH_VARARGS|METH_KEYWORDS, 
		"normalize(corrl, bins, timesT=None, timesU=None, edges=None, norm_bin_width=True, minzero=False, validate=True)\n"
		"--\n\n"
		"Normalize raw correlation curve according to bins and photons.\n"
		"Should only be performed in normalize was False or 'hist' in \n"
		"call to correlate. Set norm_bin_width to False to not normalize\n"
		"for the bin widths, but only the photons. If timesT and timesU\n"
		"are not supplied, then will only normalize base on bin width.\n"
		"\n"
		"Parameters\n"
		"----------\n"
		"corrl: numpy.ndarray\n"
		"    Array (1 or 2D) last dim representing the correlation,\n"
		"    other, if supplied for different correlations (for\n"
		"    instance different weights arrays).\n"
		"bins: numpy.ndarray\n"
		"timesT/timesU: list[numpy.ndarray]|numpy.ndarray\n"
		"    Arrival times of photons to cross-correlate.\n"
		"    Must be integer array\n"
		"edges: numpy.ndarray\n"
		"    1 or 2D array, deffining the start/stop times of each burst.\n"
		"    The last dimension must have size = 2.\n"
		""
		"minzero: bool, optional\n"
		"    When using automatic assement of start/stop times for normalization\n"
		"    True will make the start time always 0, False (default),\n"
		"    then use the first photon in the times array(s) for normalization\n"
		"validate: bool, optional\n"
		"    Whether or not to perform a check that times arrays are\n"
		"    monotonically increasing, if True (default), will raise an\n"
		"    error if photons are not monotinically increasing. If false\n"
		"    no error will be raised, but if photons are no monotonically\n"
		"    increasing, the results will be non-defined. Default is True.\n"
		"Returns\n"
		"-------\n"
		"normcorrl: numpy.ndarray\n"
		"    normalized array, of same shape as corrl\n"
	},
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
