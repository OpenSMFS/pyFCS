#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#include "correlate.h"

// sub-part of nplist, identifies what type of array it contains
typedef enum
{
	DTYPE_UNDEFINED,
	DTYPE_TIMES,
	DTYPE_NANOS,
	DTYPE_WEIGHTSLIST,
	DTYPE_WEIGHTSINDEX
} ArrayType;

// flag for identifying which casting function to use in casting numpy array
typedef enum
{
	CF_INT, // enusre the numpy array is an integer type before forcecasting
	CF_FLOAT // use no forcecast, array should be naturally castable to float
} CastFunc;

// 
typedef enum {
	FC_ERROR,
	FC_INT,
	FC_WEIGHTSLIST,
	FC_WEIGHTSINDEX
} FuncCode;

typedef struct
{
	size_t n;
	size_t *lens;
	cc_times **times;
} timeslist;

typedef struct
{
	size_t n;
	size_t *lens;
	cc_nanos **nanos;
} nanoslist;

typedef struct
{
	size_t n;
	size_t nW;
	int nd;
	size_t *lens;
	cc_weights ***weights;
} weightslist;

typedef struct
{
	size_t nW;
	size_t nI;
	int nd;
	cc_weights **weights;
} weightsindex;

typedef union
{
	timeslist *times;
	nanoslist *nanos;
	weightslist *weightsL;
	weightsindex *weightsI;
} ArrayData;

typedef struct
{
	size_t n;
	PyArrayObject** arrs;
	ArrayType dtype;
	ArrayData cdata;
} nplist;

typedef struct
{
	size_t n;
	PyArrayObject *npdata;
	cc_times **edges;
} Edges;

typedef struct
{
	PyArrayObject *npdata;
	size_t nbin;
	cc_times *bins;
} Bins;

// free all the data, for when timeslists is not using data owned by nplist
static inline void free_timeslist(timeslist *times)
{
	if (times != NULL)
	{
		if (times->times != NULL)
		{
			for (size_t i = 0; i < times->n; i++)
			{
				Xfree(times->times[i]);
				times->times[i] = NULL;
			}
			free(times->times);
			times->times = NULL;
		}
		Xfree(times->lens);
		times->lens = NULL;
		times->n = 0;
		free(times);
	}
}

// free when times data is owned by nplist
static inline void free_nptimeslist(timeslist *times)
{
	if (times != NULL)
	{
		Xfree(times->times);
		times->times = NULL;
		Xfree(times->lens);
		times->lens = NULL;
		times->n = 0;
		free(times);
	}
}

// free all the data, for when nanoslists is not using data owned by nplist
static inline void free_nanoslist(nanoslist *nanos)
{
	if (nanos != NULL)
	{
		if (nanos->nanos != NULL)
		{
			for (size_t i = 0; i < nanos->n; i++)
			{
				Xfree(nanos->nanos[i]);
				nanos->nanos[i] = NULL;
			}
			free(nanos->nanos);
			nanos->nanos = NULL;
		}
		Xfree(nanos->lens);
		nanos->lens = NULL;
		nanos->n = 0;
		free(nanos);
	}
}

// free when nanos data is owned by nplist
static inline void free_npnanoslist(nanoslist *nanos)
{
	if (nanos != NULL)
	{
		Xfree(nanos->nanos);
		nanos->nanos = NULL;
		Xfree(nanos->lens);
		nanos->lens = NULL;
		nanos->n = 0;
		free(nanos);
	}
}

// free weigthslist when weightslist is not owned by nplist
static inline void free_weightslist(weightslist *weights)
{
	if (weights != NULL)
	{
		if (weights->weights != NULL)
		{
			for (size_t i = 0; i < weights->n; i++)
			{
				if (weights->weights[i] != NULL)
				{
					for (size_t j = 0; j < weights->nW; j++)
					{
						Xfree(weights->weights[i][j]);
						weights->weights[i][j] = NULL;
					}
					free(weights->weights[i]);
					weights->weights[i] = NULL;
				}
			}
			free(weights->weights);
			weights->weights = NULL;
		}
		Xfree(weights->lens);
		weights->lens = NULL;
		weights->n = 0;
		free(weights);
	}
}

// free weightslist when weightslist is owned by nplist
static inline void free_npweightslist(weightslist *weights)
{
	if (weights != NULL)
	{
		if (weights->weights != NULL)
		{
			for (size_t i = 0; i < weights->n; i++)
			{
				Xfree(weights->weights[i]);
				weights->weights[i] = NULL;
			}
			free(weights->weights);
			weights->weights = NULL;
		}
		Xfree(weights->lens);
		weights->lens = NULL;
		weights->nW = 0;
		weights->n = 0;
		free(weights);
	}
}

// free weights index when it has not nplist parent
static inline void free_weightsindex(weightsindex *weights)
{
	if (weights != NULL)
	{
		if (weights->weights != NULL)
		{
			for ( size_t i = 0; i < weights->nW; i++)
			{
				Xfree(weights->weights[i]);
				weights->weights[i] = NULL;
			}
			free(weights->weights);
			weights->weights = NULL;
		}
		weights->nW = 0;
		weights->nI = 0;
		free(weights);
	}
}

// free weights index when its data is not owned by nplist
static inline void free_npweightsindex(weightsindex *weights)
{
	if (weights != NULL)
	{
		Xfree(weights->weights);
		weights->weights = NULL;
		weights->nW = 0;
		weights->nI = 0;
		free(weights);
	}
}

// free nplist, including allocated cdata
static inline void free_nplist(nplist *in)
{
	if (in != NULL)
	{
		switch(in->dtype)
		{
			case DTYPE_UNDEFINED:
				break;
			case DTYPE_TIMES:
				free_nptimeslist(in->cdata.times);
				in->cdata.times = NULL;
				break;
			case DTYPE_NANOS:
				free_npnanoslist(in->cdata.nanos);
				in->cdata.nanos = NULL;
				break;
			case DTYPE_WEIGHTSLIST:
				free_npweightslist(in->cdata.weightsL);
				in->cdata.weightsL = NULL;
				break;
			case DTYPE_WEIGHTSINDEX:
				free_npweightsindex(in->cdata.weightsI);
				in->cdata.weightsI = NULL;
				break;
			default:
				break;
		}	
		if (in->arrs != NULL)
		{
			for (size_t i = 0; i < in->n; i++)
				Py_XDECREF(in->arrs[i]);
			free(in->arrs);
			in->arrs = NULL;
		}
		in->n = 0;
		free(in);
	}
}

// free edges structure
static inline void free_edges(Edges *edges)
{
	if (edges != NULL)
	{
		if (edges->edges != NULL)
		{
			if (edges->npdata == NULL)
			{
				for (size_t i = 0; i < edges->n; i++)
				{
					Xfree(edges->edges[i]);
					edges->edges[i] = NULL;
				}
			}
			free(edges->edges);
			edges->edges = NULL;
		}
		if (edges->npdata != NULL)
		{
			Py_DECREF(edges->npdata);
			edges->npdata = NULL;
		}
		free(edges);
	}
}

// free bins structure
static inline void free_bins(Bins *bins)
{
	if (bins != NULL)
	{
		Py_XDECREF(bins->npdata);
		bins->nbin = 0;
		bins->bins = NULL;
		free(bins);
	}
}

// added safety, like PyArray_Check, but doesn't segfault on NULLs
static inline int npArray_Check(PyObject *in)
{
	int ret;
	if (in == NULL)
		ret = FALSE;
	else
	{
		ret = PyArray_Check((PyObject*) in);
	}
	return ret;
}

// added safety, like PyArray_ISOBJECT, but doesn't segfault on NULLs
static inline int npArray_IsObject(PyObject *in)
{
	int ret;
	if (!npArray_Check(in))
		ret = FALSE;
	else
		ret = PyArray_ISOBJECT((PyArrayObject*) in);
	return ret;
}

// added safety, like PyArray_ISINTEGER, but doesn't segfault on NULLs
static inline int npArray_IsInteger(PyObject *in)
{
	int ret;
	if (!npArray_Check(in))
		ret = FALSE;
	else
		ret = PyArray_ISINTEGER((PyArrayObject*) in);
	return ret;
}

// cast to array, with requirement that input array is integer (for casting to unsigned without requiring input to be unsigned)
static inline PyArrayObject* npArray_frominteger(PyObject *in, int typenum, int min, int max, int requirements)
{
	PyObject* temp = PyArray_FROM_O(in);
	PyObject *Otemp = NULL;
	PyArrayObject *out = NULL;
	if (npArray_IsInteger(temp))
	{
		Otemp = PyArray_FROMANY(temp, typenum, min, max, requirements);
	}
	Py_XDECREF(temp);
	if (npArray_Check(Otemp))
	{
		out = (PyArrayObject*) Otemp;
	}
	else
	{
		Py_XDECREF(Otemp);
	}
	return out;
}

// cast to numpy array, just to make sure output is numpy array
static inline PyArrayObject* npArray_fromany(PyObject *in, int typenum, int min, int max, int requirements)
{
	PyArrayObject *out = NULL;
	PyObject* temp = PyArray_FROMANY(in, typenum, min, max, requirements);
	if (npArray_Check(temp))
		out = (PyArrayObject*) temp;
	else
		Py_XDECREF(temp);
	return out;
}

static inline PyArrayObject* npArray_fromcode(PyObject *in, CastFunc cast_func, int typenum, int min, int max, int requirements)
{
	PyArrayObject *out = NULL;
	switch(cast_func)
	{
		case CF_INT:
		out = npArray_frominteger(in, typenum, min, max, requirements);
		break;
		case CF_FLOAT:
		out = npArray_fromany(in, typenum, min, max, requirements);
		break;
	}
	return out;
}

// make nplist from input that has been cast to array, when input looks like single array
// error codes
// -3: memory error
// -1: incorect type
//  0: successfully cast
static inline int allocate_nplist_single(PyObject* in, CastFunc cast_func, int typenum, int min, int max, int requirements, nplist **out)
{
	int ret = 0;
	nplist *outlist = NULL;
	PyArrayObject *temp = npArray_fromcode(in, cast_func, typenum, min, max, requirements);
	if (temp == NULL)
	{
		ret = -1;
		goto final;
	}
	if ((outlist = calloc(1, sizeof(nplist))) == NULL)
	{
		Py_DECREF(temp);
		temp = NULL;
		ret = -3;
		goto final;
	}
	outlist->dtype = DTYPE_UNDEFINED;
	if ((outlist->arrs = calloc(1, sizeof(PyArrayObject*))) == NULL)
	{
		Py_DECREF(temp);
		temp = NULL;
		free(outlist);
		outlist = NULL;
		ret = -3;
		goto final;
	}
	outlist->arrs[0] = temp;
	outlist->n = 1;
	*out = outlist;
	final:
	return ret;
}

// make nplist from input that has been cast to array when array type is object (array sequence)
// error codes
// -5: problem in C code
// -4: bad internal call
// -3: memory error
// -2: 0 size array
// -1: incorect type
//  0: successfully cast
// positive: error converting array
static inline int allocate_nplist_multiple(PyObject* in, CastFunc cast_func, int typenum, int min, int max, int requirements, nplist **out)
{
	int ret = 0;
	Py_ssize_t len = 0;
	PyObject *temp = NULL;
	nplist *larray = NULL;
	if (!PySequence_Check(in))
	{
		ret = -5;
		goto final;
	}
	if ((len = PySequence_Size(in)) < 1)
	{
		ret = (len < 0) ? -4 : -2;
		goto final;
	}
	if ((larray  = calloc(1, sizeof(nplist))) == NULL)
	{
		ret = -3;
		goto final;
	}
	if ((larray->arrs = calloc(len, sizeof(PyArrayObject*))) == NULL)
	{
		free(larray);
		larray = NULL;
		ret = -3;
		goto final;
	}
	larray->n = (size_t) len;
	larray->dtype = DTYPE_UNDEFINED;
	for (Py_ssize_t i = 0; i < len; i++)
	{
		if ((temp = PySequence_GetItem(in, i)) == NULL)
		{
			ret = (int) i + 1;
			break;
		}
		larray->arrs[i] = npArray_fromcode(temp, cast_func, typenum, min, max, requirements);
		Py_DECREF(temp);
		if (larray->arrs[i] == NULL)
		{
			ret = (int) i + 1;
			break;
		}
	}
	if (ret != 0)
	{
		free_nplist(larray);
		larray = NULL;
	}
	else
		*out = larray;
	final:
	return ret;
}

// check if can be interpreted as a nested sequence
static inline int is_arrayorsequence(PyObject *in)
{
	int ret = -1;
	if (in == NULL)
		goto final;
	if (PyArray_Check(in))
	{
		if (PyArray_ISOBJECT((PyArrayObject*) in))
			ret = 1;
		else
			ret = 0;
	}
	else if (PySequence_Check(in))
	{
		PyObject *temp = PySequence_GetItem(in, 0);
		if (temp != NULL)
		{
			if (PySequence_Check(temp))
				ret = 1;
			else
				ret = 0;
			Py_DECREF(temp);
		}
	}
	final:
	return ret;
}

// error codes:
// -5: problem in C code
// -4: bad internal call
// -3: memory error
// -2: zero size array
// -1: not able to convert to numpy array
static inline int cast_sequence(PyObject *in, CastFunc cast_func, int typenum, int min, int max, int requirements, nplist **out)
{
	int ret = 0;
	int code = is_arrayorsequence(in);
	switch (code)
	{
		case 0:
			ret = allocate_nplist_single(in, cast_func, typenum, min, max, requirements, out);
			break;
		case 1:
			ret = allocate_nplist_multiple(in, cast_func, typenum, min, max, requirements, out);
			break;
		default:
			ret = -1;
			break;
	}
	return ret;
}

static inline int cast_sequence_weight(PyObject *in, CastFunc cast_func, int typenum, int min, int max, int requirements, size_t n, nplist **out)
{
	int ret = 0;
	int code = is_arrayorsequence(in);
	switch (code)
	{
		case 0:
			ret = allocate_nplist_single(in, cast_func, typenum, min, max, requirements, out);
			break;
		case 1:
			if (n != (size_t) PySequence_Length(in))
				ret = allocate_nplist_single(in, cast_func, typenum, min, max, requirements, out);
			else
				ret = allocate_nplist_multiple(in, cast_func, typenum, min, max, requirements, out);
			break;
		default:
			ret = -1;
			break;
	}
	return ret;
}

static inline int err_sequence(Py_ssize_t err, const char *name, const char *descr)
{
	int ret = FALSE;
	switch(err)
	{
		case -7:
			ret = TRUE;
			PyErr_Format(PyExc_ValueError, "%s cannot have 0 length in 0th dimension", name);
			break;
		case -6:
			ret = TRUE;
			PyErr_Format(PyExc_RuntimeError, "error in C code, converted list %s failed to allocate", name);
			break;
		case -5:
			ret = TRUE;
			PyErr_Format(PyExc_RuntimeError, "error in C code converting %s", name);
			break;
		case -4: // bad internal call by PyArray or Py___ function, let it set the error
			ret = TRUE;
			break;
		case -3:
			ret = TRUE;
			PyErr_NoMemory();
			break;
		case -2:
			ret = TRUE;
			PyErr_Format(PyExc_TypeError, "%s cannot be empty sequence", name);
			break;
		case -1:
			ret = TRUE;
			PyErr_Format(PyExc_TypeError, "%s cannot be interpreted as %s or sequence thereof", name, descr);
			break;
		case 0:
			ret = FALSE;
			break;
		default:
			ret = TRUE;
			if (err > 0)
				PyErr_Format(PyExc_TypeError, "%s[%d] not interpretable as %s array", name, err - 1, descr);
			else
				PyErr_SetString(PyExc_SystemError, "Error in pyFCS C code, produced unexpected error code\n");
			break;
	}
	return ret;
}


// error codes
//  -7: 0 length weights
//  -6: unallocated nplist
//  -4: negative dimension, error set by function
//  -3: no memory
//  -2: not 1D array (or 1/2D if weights)
//  -1: wrong type
//   0: Success
// positive: error in array i - 1
static inline int set_timesdata(nplist *npdata)
{
	int ret = 0;
	if (npdata->n == 0)
	{
		ret = -6;
		goto final;
	}
	// check array types and dimensions are valid
	if ((npdata->cdata.times = calloc(1, sizeof(timeslist))) == NULL)
	{
		ret = -3;
		goto final;
	}
	if ((npdata->cdata.times->lens = calloc(npdata->n, sizeof(size_t))) == NULL)
	{
		ret = -3;
		free(npdata->cdata.times);
		npdata->cdata.times = NULL;
		goto final;
	}
	if ((npdata->cdata.times->times = calloc(npdata->n, sizeof(cc_times*))) == NULL)
	{
		ret = -3;
		free(npdata->cdata.times->lens);
		npdata->cdata.times->lens = NULL;
		free(npdata->cdata.times);
		npdata->cdata.times = NULL;
		goto final;
	}
	npdata->cdata.times->n = npdata->n;
	npdata->dtype = DTYPE_TIMES;
	npy_intp nplen;
	for (size_t i = 0; i < npdata->n; i++)
	{
		if (PyArray_TYPE(npdata->arrs[i]) != NPY_UINT64)
		{
			ret = i + 1;
			break;
		}
		if (PyArray_NDIM(npdata->arrs[i]) != 1)
		{
			ret = i + 1;
			break;
		}
		if ((nplen = PyArray_DIM(npdata->arrs[i], 0)) < 0)
		{
			ret = i + 1;
			break;
		}
		npdata->cdata.times->lens[i] = (size_t) nplen;
		npdata->cdata.times->times[i] = (cc_times*) PyArray_DATA(npdata->arrs[i]);
	}
	if (ret != 0)
	{
		free_nptimeslist(npdata->cdata.times);
		npdata->cdata.times = NULL;
		npdata->dtype = DTYPE_UNDEFINED;
	}
	final:
	return ret;
}

// set cdata for nanos nplist
static inline int set_nanosdata(nplist *npdata)
{
	int ret = 0;
	if (npdata->n == 0)
	{
		ret = -6;
		goto final;
	}
	// check array types and dimensions are valid
	if ((npdata->cdata.nanos = calloc(1, sizeof(nanoslist))) == NULL)
	{
		ret = -3;
		goto final;
	}
	if ((npdata->cdata.nanos->lens = calloc(npdata->n, sizeof(size_t))) == NULL)
	{
		ret = -3;
		free(npdata->cdata.nanos);
		npdata->cdata.nanos = NULL;
		goto final;
	}
	if ((npdata->cdata.nanos->nanos = calloc(npdata->n, sizeof(cc_nanos*))) == NULL)
	{
		ret = -3;
		free(npdata->cdata.nanos->lens);
		npdata->cdata.nanos->lens = NULL;
		free(npdata->cdata.nanos);
		npdata->cdata.nanos = NULL;
		goto final;
	}
	npdata->cdata.nanos->n = npdata->n;
	npdata->dtype = DTYPE_NANOS;
	npy_intp nplen;
	for (size_t i = 0; i < npdata->n; i++)
	{
		if (PyArray_TYPE(npdata->arrs[i]) != NPY_UINT16)
		{
			ret = i + 1;
			break;
		}
		if (PyArray_NDIM(npdata->arrs[i]) != 1)
		{
			ret = i + 1;
			break;
		}
		if ((nplen = PyArray_DIM(npdata->arrs[i], 0)) < 0)
		{
			ret = i + 1;
			break;
		}
		npdata->cdata.nanos->lens[i] = (size_t) nplen;
		npdata->cdata.nanos->nanos[i] = (cc_nanos*) PyArray_DATA(npdata->arrs[i]);
	}
	if (ret != 0)
	{
		free_npnanoslist(npdata->cdata.nanos);
		npdata->cdata.nanos = NULL;
		npdata->dtype = DTYPE_UNDEFINED;
	}
	final:
	return ret;
}

// assign the ->cdata of nplist as weightsL array, assigns appropriate pointers
static inline int set_weightslistdata(nplist *npdata)
{
	int ret = 0, nd = 0, tnd;
	npy_intp npW, nplen, tnpW; // dimensions of arrays
	if (npdata->n == 0)
	{
		ret = -6;
		goto final;
	}
	nd = PyArray_NDIM(npdata->arrs[0]);
	if ( (nd < 1) || (nd > 2) )
	{
		ret = (nd < 0) ? -4 : -2;
		goto final;
	}
	npW = (nd == 1) ? 1 : PyArray_DIM(npdata->arrs[0], 0);
	// check that all dimensions are the same
	if (npW < 1)
	{
		ret = (npW < 0) ? -4 : -7;
		goto final;
	}
	// check consistency of each data's 0th dimension
	if ((npdata->cdata.weightsL = calloc(1, sizeof(weightslist))) == NULL)
	{
		ret = -3;
		goto final;
	}
	if ((npdata->cdata.weightsL->lens = calloc(npdata->n, sizeof(size_t))) == NULL)
	{
		free(npdata->cdata.weightsL);
		npdata->cdata.weightsL = NULL;
		ret = -3;
		goto final;
	}
	if ((npdata->cdata.weightsL->weights = calloc(npdata->n, sizeof(cc_weights**))) == NULL)
	{
		free(npdata->cdata.weightsL->lens);
		npdata->cdata.weightsL = NULL;
		free(npdata->cdata.weightsL);
		npdata->cdata.weightsL = NULL;
		ret = -3;
		goto final;
	}
	npdata->cdata.weightsL->nd = nd;
	npdata->cdata.weightsL->n = npdata->n;
	npdata->cdata.weightsL->nW = (size_t) npW;
	npdata->dtype = DTYPE_WEIGHTSLIST;
	for ( size_t i = 0; i < npdata->n; i++)
	{
		if (PyArray_TYPE(npdata->arrs[i]) != NPY_DOUBLE)
		{
			ret = i + 1;
			break;
		}
		tnd = PyArray_NDIM(npdata->arrs[i]);
		if ((tnd < 1) || (tnd > 2))
		{
			ret = i + 1;
			break;
		}
		tnpW = (tnd == 1) ? 1 : PyArray_DIM(npdata->arrs[i], 0);
		if (tnpW != npW)
		{
			ret = (tnpW < 0) ? -4 : (int) i + 1;
			break;
		}
		if ((nplen = PyArray_DIM(npdata->arrs[i], PyArray_NDIM(npdata->arrs[i]) - 1)) < 0)
		{
			ret = (nplen < 0) ? -4 : (int) i + 1;
			break;
		}
		npdata->cdata.weightsL->lens[i] = (size_t) nplen;
		if ((npdata->cdata.weightsL->weights[i] = calloc(npdata->cdata.weightsL->nW, sizeof(cc_weights*))) == NULL)
		{
			ret = -3;
			break;
		}
		npdata->cdata.weightsL->weights[i][0] = PyArray_DATA(npdata->arrs[i]);
		for (size_t j = 1; j < npdata->cdata.weightsL->nW ; j++)
			npdata->cdata.weightsL->weights[i][j] = npdata->cdata.weightsL->weights[i][0] + (j*nplen);
	}
	if ( ret != 0)
	{
		free_weightslist(npdata->cdata.weightsL);
		npdata->cdata.weightsL = NULL;
		npdata->dtype = DTYPE_UNDEFINED;
	}
	final:
	return ret;
}

// assign the ->cdata of nplist as weightsI array, assigns appropriate pointers
static inline int set_weightsindexdata(nplist *npdata)
{
	int ret = 0, nd = 0;
	npy_intp npW, npI;
	if (npdata->n != 1)
	{
		ret = (npdata->n == 0) ? -6 : -2;
		goto final;
	}
	if (PyArray_TYPE(npdata->arrs[0]) != NPY_DOUBLE)
	{
		ret = -1;
		goto final;
	}
	nd = PyArray_NDIM(npdata->arrs[0]); 
	if ((nd < 1) || (nd > 2))
	{
		ret = (nd < 0) ? -4 : -2;
		goto final;
	}
	npW = (nd == 1) ? 1 : PyArray_DIM(npdata->arrs[0], 0);
	if (npW < 1)
	{
		ret = (npW == 0) ? -7 : -4;
		goto final;
	}
	npI = PyArray_DIM(npdata->arrs[0], nd - 1);
	if (npI < 1)
	{
		ret = (npI == 0) ? -7 : -4;
		goto final;
	}
	if ((npdata->cdata.weightsI = calloc(1, sizeof(weightsindex))) == NULL)
	{
		ret = -3;
		goto final;
	}
	if ((npdata->cdata.weightsI->weights = calloc(npW, sizeof(double*))) == NULL)
	{
		free(npdata->cdata.weightsI);
		npdata->cdata.weightsI = NULL;
		goto final;
	}
	npdata->cdata.weightsI->nW = (size_t) npW;
	npdata->cdata.weightsI->nI = (size_t) npI;
	npdata->cdata.weightsI->nd = nd;
	npdata->dtype = DTYPE_WEIGHTSINDEX;
	npdata->cdata.weightsI->weights[0] = PyArray_DATA(npdata->arrs[0]);
	for (size_t i = 1; i < npdata->cdata.weightsI->nW; i++)
		npdata->cdata.weightsI->weights[i] = npdata->cdata.weightsI->weights[0] + (i * npI);
	final:
	return ret;
}

// check if the values in a timeslist are monotonically increasing, returns true if not monotonically increasing
static inline int is_monotonicincrease(timeslist *times, const char*name)
{
	int ret = FALSE;
	for (size_t i = 0; i < times->n; i++)
	{
		for (size_t j = 0, jj= 1; jj < times->lens[i]; j++, jj++)
		{
			if (times->times[i][j] > times->times[i][jj])
			{
				if (times->n == 1) // error for single array correlation
					PyErr_Format(PyExc_ValueError, "%s is not monotonically increasesing, (error at element %lu)", name, jj);
				else // error for multi-list correlation
					PyErr_Format(PyExc_ValueError, "%s is not monotonically increasesing, (error in array %lu, element %lu)", name, i, jj);
				ret = TRUE;
				break;
			}
		}
		if (ret)
			break;
	}
	return ret;
}

// check if edges are larger than arrays in timesT/U
static inline int check_edges(Edges *edges, timeslist *timesT, timeslist *timesU)
{
	int err = FALSE;
	for (size_t i = 0; i < edges->n; i++)
	{
		cc_times min = edges->edges[i][0];
		cc_times max = edges->edges[i][1];
		if ((min > timesT->times[i][0]) || (min > timesU->times[i][0]))
		{
			if (timesT->n == 1)
				PyErr_SetString(PyExc_ValueError, "Invalid edges, timesT/U have elements less than min edge");
			else
				PyErr_Format(PyExc_ValueError, "Invalid edges, timesT/U[%ld] have elements less than min edge", i);
			err = TRUE;
			break;
		}
		if ((max < timesT->times[i][timesT->lens[i]-1]) || (max < timesU->times[i][timesU->lens[i]-1]))
		{
			if (timesT->n == 1)
				PyErr_SetString(PyExc_ValueError, "Invalid edges, timesT/U have elements greater than max edge");
			else
				PyErr_Format(PyExc_ValueError, "Invalid edges, timesT/U[%ld] have elements greater than max edge", i);
			err = TRUE;
			break;
		}
	}
	return err;
}

// allocate edges 
static inline Edges* get_edges_fromnp(PyObject *pyedges, timeslist *timesT, timeslist *timesU, int validate)
{
	Edges *edges = NULL;
	int err = FALSE;
	int ndim;
	size_t i;
	if (timesT->n != timesU->n)
	{
		PyErr_Format(PyExc_ValueError, "Number of arrays in timesT/U be the same, got %lu and %lu respectively", timesT->n, timesU->n);
		goto exit;
	}
	if ((edges = calloc(1, sizeof(Edges))) == NULL)
	{
		PyErr_NoMemory();
		goto exit;
	}
	if ((edges->npdata = npArray_frominteger(pyedges, NPY_UINT64, 1, 2, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_FORCECAST)) == NULL)
	{
		PyErr_SetString(PyExc_TypeError, "edges must be 2 element Nx2 element 1/2D integer array");
		err = TRUE;
		goto frees;
	}
	ndim = PyArray_NDIM(edges->npdata);
	if ((ndim < 1)||(2 < ndim))
	{
		PyErr_SetString(PyExc_ValueError, "edges must be 2 element Nx2 element 1/2D integer array");
		err = TRUE;
		goto frees;
	}
	if (PyArray_DIM(edges->npdata, ndim - 1) != 2)
	{
		PyErr_Format(PyExc_ValueError, "edges must be 2 element Nx2 element 1/2D integer array, got last dim size %ld", PyArray_DIM(edges->npdata, ndim - 1));
		err = TRUE;
		goto frees;
	}
	if (ndim == 2)
	{
		npy_intp check_len = PyArray_DIM(edges->npdata, 0);
		if (check_len < 0) // invlaid array, let numpy set the error
		{
			err = TRUE;
			goto frees;
		}
		if ((size_t) check_len != timesT->n)
		{
			PyErr_Format(PyExc_ValueError, "edges has wrong first dimenstion from timesT/U, expected %lu, but edges had %ld", timesT->n, PyArray_DIM(edges->npdata, 0));
			err = TRUE;
			goto frees;
		}
	}
	edges->n = timesT->n;
	if ((edges->edges = calloc(timesT->n, sizeof(cc_times*))) == NULL)
	{
		PyErr_NoMemory();
		err = TRUE;
		goto frees;
	}
	edges->edges[0] = PyArray_DATA(edges->npdata);
	switch (ndim)
	{
		case 1:
			for (i = 1; i < timesT->n; i++)
				edges->edges[i] = edges->edges[0];
			break;
		case 2:
			for (i = 1; i < timesT->n; i++)
				edges->edges[i] = edges->edges[0] + (2*i);
			break;
	}
	if (validate)
	{
		if (check_edges(edges, timesT, timesU))
			err = TRUE;
	}
	frees:
	if (err)
	{
		free_edges(edges);
		edges = NULL;
	}
	exit:
	return edges;
}

static inline Edges* get_edges_fromtimes(timeslist *timesT, timeslist *timesU, int minzero)
{
	Edges *edges = NULL;
	if (timesT->n != timesU->n)
	{
		PyErr_Format(PyExc_ValueError, "mismatched nubmer of arrays in timesT/U, got %lu and %lu respectively", timesT->n, timesU->n);
		goto final;
	}
	if ((edges = calloc(1, sizeof(Edges))) == NULL)
	{
		PyErr_NoMemory();
		goto final;
	}
	if ((edges->edges = malloc(timesT->n * sizeof(cc_times*))) == NULL)
	{
		PyErr_NoMemory();
		free(edges);
		edges = NULL;
		goto final;
	}
	for (size_t i = 0; i < timesT->n; i++)
	{
		if ((edges->edges[i] = malloc( 2 * sizeof(cc_times))) == NULL)
		{
			PyErr_NoMemory();
			for (size_t j = 0; j < i; j++)
			{
				free(edges->edges[j]);
				edges->edges[j] = NULL;
			}
			free(edges->edges);
			edges->edges = NULL;
			free(edges);
			edges = NULL;
			break;
		}
		if (minzero)
			edges->edges[i][0] = 0;
		else
			edges->edges[i][0] = (timesT->times[i][0] < timesU->times[i][0]) ? timesT->times[i][0] : timesU->times[i][0];
		edges->edges[i][1] = (timesT->times[i][timesT->lens[i]-1] > timesU->times[i][timesU->lens[i]-1]) ? timesT->times[i][timesT->lens[i]-1] : timesU->times[i][timesU->lens[i]-1];
	}
	edges->n = timesT->n;
	final:
	return edges;
}

static inline Edges* cast_edges(PyObject *pyedges, timeslist *timesT, timeslist *timesU, int minzero, int validate)
{
	Edges *edges = NULL;
	if ((pyedges == NULL) || ( pyedges == Py_None))
	{
		edges = get_edges_fromtimes(timesT, timesU, minzero);
	}
	else
	{
		if (minzero)
			PyErr_WarnEx(PyExc_UserWarning, "edges specified, ignoring minzero argument", 1);
		edges = get_edges_fromnp(pyedges, timesT, timesU, validate);
	}
	return edges;
}

static inline Bins* cast_bins(PyObject *pybins, int validate)
{
	PyArrayObject *npbins = NULL;
	npy_intp nbin;
	int err = FALSE;
	Bins *bins = NULL;
	cc_times *binptr = NULL;
	if ((npbins = npArray_frominteger(pybins, NPY_UINT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_FORCECAST)) == NULL)
	{
		PyErr_SetString(PyExc_TypeError, "bins must be 1D integer array");
		goto final;
	}
	if (PyArray_NDIM(npbins) != 1)
	{
		PyErr_Format(PyExc_ValueError, "bins must be 1D integer array with 2 or more elements, got %d dimensions", PyArray_NDIM(npbins));
		Py_DECREF(npbins);
		goto final;
	}
	if ((nbin = PyArray_DIM(npbins, 0)) < 2)
	{
		if (nbin >= 0)
			PyErr_Format(PyExc_ValueError, "bins must be 1D array with 2 or more elements, got %ld elements", nbin);
		Py_DECREF(npbins);
		goto final;
	}
	binptr = PyArray_DATA(npbins);
	if (validate)
	{
		for (npy_intp i = 0, ii = 1; ii < nbin; i++, ii++)
		{
			if (binptr[i] > binptr[ii])
			{
				err = TRUE;
				PyErr_Format(PyExc_ValueError, "bins must be 1D array with 2 or more monotinically increasing elements, error in position %ld", i);
				Py_DECREF(npbins);
				break;
			}
		}
		if (err)
			goto final;
	}
	if ((bins = calloc(1, sizeof(Bins))) == NULL)
	{
		PyErr_NoMemory();
		Py_DECREF(npbins);
		goto final;
	}
	bins->npdata = npbins;
	bins->nbin = (size_t) nbin;
	bins->bins = binptr;
	final:
	return bins;
}

static inline int is_allequal(size_t n, size_t *lensA, size_t *lensB)
{
	int ret = TRUE;
	for (size_t i = 0; i < n; i++)
	{
		if (lensA[i] != lensB[i])
		{
			ret = FALSE;
			break;
		}
	}
	return ret;
}

static inline int check_weightslist(timeslist *timesT, timeslist *timesU, weightslist *weightsT, weightslist *weightsU)
{
	int err = FALSE;
	if (timesT->n != timesU->n) // check same number of arrays (bursts) in timesT/U
	{
		PyErr_Format(PyExc_ValueError, "timesT/U must contain same number of arrrays, got %lu and %lu respectively", timesT->n, timesU->n);
		err = TRUE;
		goto final;
	}
	if (weightsT->n != weightsU->n) // check same number of arrays (bursts) in weightsT/U
	{
		PyErr_Format(PyExc_ValueError, "Inconsistent number of arrays in weightsT/U, got %lu and %lu respectively, must match timesT/U, for which there were %lu", 
						weightsT->n, weightsU->n, timesT->n);
		err = TRUE;
		goto final;
	}
	if (timesT->n != weightsT->n) // check times and weights have same number of arrays (bursts)
	{
		PyErr_Format(PyExc_ValueError, "Inconsistent number of arrays in timesT/U and weightsT/U, got %lu and %lu respectively", timesT->n, weightsT->n);
		err = TRUE;
		goto final;
	}
	if (!is_allequal(timesT->n, timesT->lens, weightsT->lens)) // check that weights and times T arrays have same number of photons
	{
		PyErr_SetString(PyExc_ValueError, "one or more arrays in timesT and weigthsT have inconsistent lengths");
		err = TRUE;
		goto final;
	}
	if (!is_allequal(timesU->n, timesU->lens, weightsU->lens)) // check that weigths and times U arrays have same number of photons
	{
		PyErr_SetString(PyExc_ValueError, "one or more arrays in timesU and weigthsU have inconsistent lengths");
		err = TRUE;
	}
	final:
	return err;
}

static inline int get_weightslist_ndim(timeslist *timesT, timeslist *timesU, weightslist *weightsT, weightslist *weightsU, int cross_correlate)
{
	if (check_weightslist(timesT, timesU, weightsT, weightsU))
		return -1;
	int outndim = 0;
	if (cross_correlate)
	{
		outndim = weightsT->nd + weightsU->nd - 1;
		// formula for number of output dimensions based on dims of 
		// weights ndim. To understand the formula, the number of output
		// dimensions has 1 dim for the bins, and for each weights, take
		// 1 dim off of its number of dims for the base photons, so the output
		// becomes 1 + (nd_t -1) + (nd_u - 1) ~ nd_t - 1 
	}
	else  // if not cross-correlating, then weightsT/U must have matching 0th dim
	{
		if (weightsT->nW != weightsU->nW)
		{
			PyErr_Format(PyExc_ValueError, "Inconsistent number of weight arrays (size of dimension 0 of weights arrays) in weightsT/U, got %lu and %lu respectively", weightsT->nW, weightsU->nW);
			outndim = -1;
		}
		else
			outndim = weightsT->nd; // if the number of weights matches, then the output nd is same as that of weights
	}
	return outndim;
}

static inline cc_nanos max_nano(nanoslist *nanos)
{
	cc_nanos max = 0;
	for (size_t i = 0; i < nanos->n; i++)
	{
		for (size_t j = 0; j < nanos->lens[i]; j++)
		{
			if ( nanos->nanos[i][j] > max)
				max = nanos->nanos[i][j];
		}
	}
	return max;
}

static inline int check_weightsnanos(timeslist *timesT, timeslist *timesU, weightsindex *weightsT, weightsindex *weightsU, nanoslist *nanosT, nanoslist *nanosU)
{
	int err = FALSE;
	if (timesT->n != timesU->n)
	{
		PyErr_Format(PyExc_ValueError, "timesT/U must contain same number of arrays, got %lu and %lu respectively", timesT->n, timesU->n);
		err = TRUE;
		goto final;
	}
	if (nanosT->n != nanosU->n)
	{
		PyErr_Format(PyExc_ValueError, "nanosT/U must contain same number of arrays, got %lu and %lu respectively", timesT->n, timesU->n);
		err = TRUE;
		goto final;
	}
	if (nanosT->n != timesT->n)
	{
		PyErr_Format(PyExc_ValueError, "Inconsistent number of arrays in nanosT/U, got %lu and %lu respectively, must match timesT/U, for which there were %lu", 
						nanosT->n, nanosU->n, timesT->n);
		err = TRUE;
		goto final;
	}
	if (!is_allequal(timesT->n, timesT->lens, nanosT->lens))
	{
		PyErr_SetString(PyExc_ValueError, "one or more arrays in timesT and nanosT have inconsistent lengths");
		err = TRUE;
		goto final;
	}
	if (!is_allequal(timesU->n, timesU->lens, nanosU->lens))
	{
		PyErr_SetString(PyExc_ValueError, "one or more arrays in timesT and nanosT have inconsistent lengths");
		err = TRUE;
		goto final;
	}
	if (weightsT->nI <= max_nano(nanosT))
	{
		PyErr_Format(PyExc_ValueError, "nanosT contains index (%lu) greater than max index allowed by weightsT (size %lu)", max_nano(nanosT), weightsT->nI );
		err = TRUE;
		goto final;
	}
	if ( weightsU->nI <= max_nano(nanosU))
	{
		PyErr_Format(PyExc_ValueError, "nanosU contains index (%lu) greater than max index allowed by weightsU (size %lu)", max_nano(nanosU), weightsU->nI );
		err = TRUE;
	}
	final:
	return err;
}
static inline int get_weightsnanos_ndim(timeslist *timesT, timeslist *timesU, weightsindex *weightsT, weightsindex *weightsU, nanoslist *nanosT, nanoslist *nanosU, int cross_correlate)
{
	if (check_weightsnanos(timesT, timesU, weightsT, weightsU, nanosT, nanosU))
		return -1;
	int outndim = 0;
	if (cross_correlate)
	{
		outndim = weightsT->nd + weightsU->nd - 1;
	}
	else 
	{
		if (weightsT->nW != weightsU->nW)
		{
			PyErr_Format(PyExc_ValueError, "Inconsistent number of weight arrays (0th dim of weights arrays) in weightsT/U, got %lu and %lu respectively", weightsT->nW, weightsU->nW);
			outndim = -1;
		}
		else
		{
			outndim = weightsT->nd;
		}
	}
	return outndim;
}


static inline void base_frees(nplist *nptimesT, nplist *nptimesU, Bins *bins, Edges *edges)
{
	free_edges(edges);
	free_bins(bins);
	free_nplist(nptimesU);
	free_nplist(nptimesT);
}

static inline void all_frees(nplist *nptimesT, nplist *nptimesU, Bins *bins, Edges *edges, nplist *npweightsT, nplist *npweightsU, nplist *npnanosT, nplist *npnanosU)
{
	free_nplist(npnanosU);
	npnanosU = NULL;
	free_nplist(npnanosT);
	npnanosT = NULL;
	free_nplist(npweightsU);
	npweightsU = NULL;
	free_nplist(npweightsT);
	npweightsT = NULL;
	base_frees(nptimesT, nptimesU, bins, edges);
	nptimesT = NULL;
	nptimesU = NULL;
	bins = NULL;
	edges = NULL;
}

static inline int base_casts(PyObject *pytimesT, PyObject *pytimesU, PyObject *pybins, PyObject *pyedges, 
						int minzero, int validate, nplist **onptimesT, nplist **onptimesU, Bins **obins, Edges **oedges)
{
	int err = FALSE;
	Py_ssize_t cerr = 0;
	nplist *nptimesT = NULL, *nptimesU = NULL;
	Bins *bins = NULL;
	Edges *edges = NULL;
	// convert timesT/U
	if ((cerr = cast_sequence(pytimesT, CF_INT, NPY_UINT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_FORCECAST, &nptimesT)) != 0)
	{
		err_sequence(cerr, "timesT", "1D array of integers");
		err = TRUE;
		goto exit;
	}
	if ((cerr = set_timesdata(nptimesT)) != 0)
	{
		err_sequence(cerr, "timesT", "1D array of integers");
		err = TRUE;
		goto frees;
	}
	if ((cerr = cast_sequence(pytimesU, CF_INT, NPY_UINT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_FORCECAST, &nptimesU)) != 0)
	{
		err_sequence(cerr, "timesU", "1D array of integers");
		err = TRUE;
		goto frees;
	}
	if ((cerr = set_timesdata(nptimesU)) != 0)
	{
		err_sequence(cerr, "timesU", "1D array of integers");
		err = TRUE;
		goto frees;
	}
	// make sure timesT/U matching number of arrays
	if (nptimesT->n != nptimesU->n)
	{
		PyErr_Format(PyExc_ValueError, "timesT/U must contain the same number of 1D integer arrays, got %lu and %lu respectively", nptimesT->n, nptimesU->n);
		err = TRUE;
		goto frees;
	}
	// check that timesT/U are monotonically increasing
	if (validate)
	{
		if (is_monotonicincrease(nptimesT->cdata.times, "timesT"))
		{
			err = TRUE;
			goto frees;
		}
		if (is_monotonicincrease(nptimesU->cdata.times, "timesU"))
		{
			err = TRUE;
			goto frees;
		}
	}
	// parse bins argument
	if ((bins = cast_bins(pybins, validate)) == NULL)
	{
		err = TRUE;
		goto frees;
	}
	// parse edges argument
	if ((edges = cast_edges(pyedges, nptimesT->cdata.times, nptimesU->cdata.times, minzero, validate)) == NULL)
	{
		err = TRUE;
		goto frees;
	}
	frees:
	if (err)
	{
		base_frees(nptimesT, nptimesU, bins, edges);
	}
	else
	{
		*oedges = edges;
		*obins = bins;
		*onptimesT = nptimesT;
		*onptimesU = nptimesU;
	}
	exit:
	return err;
}

static inline int NoneIsFalse(PyObject *in)
{
	int ret = FALSE;
	if (in != NULL)
	{
		if (in != Py_None)
			ret = TRUE;
	}
	return ret;
}

static inline FuncCode kwarg_choice(PyObject *pyweightsT, PyObject *pyweightsU, PyObject *pynanosT, PyObject *pynanosU)
{
	FuncCode ret = FC_INT;
	int wT = NoneIsFalse(pyweightsT);
	int wU = NoneIsFalse(pyweightsU);
	int nT = NoneIsFalse(pynanosT);
	int nU = NoneIsFalse(pynanosU);
	if (wT != wU)
	{
		PyErr_SetString(PyExc_TypeError, "weightsT/U must be specified together or neither");
		ret = FC_ERROR;
	}
	if (nT != nU)
	{
		PyErr_SetString(PyExc_TypeError, "nanosT/U must be specified together or neither");
		ret = FC_ERROR;
	}
	if ((ret != FC_ERROR) && wT)
	{
		ret = (nT) ? FC_WEIGHTSINDEX : FC_WEIGHTSLIST;
	}
	return ret;
}

static inline int weights_cast(PyObject *pyweightsT, PyObject *pyweightsU, nplist *nptimesT, nplist *nptimesU, nplist **npweightsT, nplist **npweightsU)
{
	int cerr = FALSE;
	if ((cerr = cast_sequence_weight(pyweightsT, CF_FLOAT, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED, nptimesT->n, npweightsT)) != 0)
	{
		err_sequence(cerr, "weightsT", "1/2D array of floats");
		goto exit;
	}
	if ((cerr = set_weightslistdata(*npweightsT)) != 0)
	{
		err_sequence(cerr, "weightsT", "1/2D array of floats");
		goto exit;
	}
	if ((cerr = cast_sequence_weight(pyweightsU, CF_FLOAT, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED, nptimesU->n, npweightsU)) != 0)
	{
		err_sequence(cerr, "weightsU", "1/2D array of floats");
		goto exit;
	}
	if ((cerr = set_weightslistdata(*npweightsU)) != 0)
		err_sequence(cerr, "weightsU", "1/2D array of floats");
	exit:
	return cerr;
}

static inline int weights_index_cast(PyObject *pyweightsT, PyObject *pyweightsU, PyObject *pynanosT, PyObject *pynanosU, nplist *nptimesT, nplist *nptimesU,
														nplist **npweightsT, nplist **npweightsU, nplist **npnanosT, nplist **npnanosU)
{
	int cerr = FALSE;
	if ((cerr = allocate_nplist_single(pyweightsT, CF_FLOAT, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED, npweightsT)) != 0)
	{
		err_sequence(cerr, "weightsT", "1/2D array of floats");
		goto exit;
	}
	if ((cerr = set_weightsindexdata(*npweightsT)) != 0)
	{
		err_sequence(cerr, "weightsT", "1/2D array of floats");
		goto exit;
	}
	if ((cerr = allocate_nplist_single(pyweightsU, CF_FLOAT, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED, npweightsU)) != 0)
	{
		err_sequence(cerr, "weightsU", "1/2D array of floats");
		goto exit;
	}
	if ((cerr = set_weightsindexdata(*npweightsU)) != 0)
	{
		err_sequence(cerr, "weightsU", "1/2D array of floats");
		goto exit;
	}
	if ((cerr = cast_sequence(pynanosT, CF_INT, NPY_UINT16, 1, 1, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_FORCECAST, npnanosT)) != 0)
	{
		err_sequence(cerr, "nanosT", "1D array of integers");
		goto exit;
	}
	if ((cerr = set_nanosdata(*npnanosT)) != 0)
	{
		err_sequence(cerr, "nanosT", "1D array of integers");
		goto exit;
	}
	if ((cerr = cast_sequence(pynanosU, CF_INT, NPY_UINT16, 1, 1, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_FORCECAST, npnanosU)) != 0)
	{
		err_sequence(cerr, "nanosU", "1D array of integers");
		goto exit;
	}
	if ((cerr = set_nanosdata(*npnanosU)) != 0)
		err_sequence(cerr, "nanosU", "1D array of integers");
	exit:
	return cerr;
}


// main function of pyFCS
static PyObject* pyFCS_correlate(PyObject* self, PyObject* args, PyObject* kwargs)
{
	char *kwlist[] = {"timesT", "timesU", "bins", "weightsT", "weightsU", "nanosT", "nanosU", "edges", "normalize", "norm_bin_width", "minzero", "cross_correlate", "validate", "max_cores", NULL};
	PyObject *pytimesT = NULL, *pytimesU, *pybins = NULL, *pyweightsT = NULL, *pyweightsU = NULL, *pynanosT = NULL, *pynanosU = NULL, *pyedges = NULL, *out = NULL;
	int normalize=TRUE, norm_bin_width=TRUE, minzero=FALSE, cross_correlate=FALSE, validate=TRUE, cerr = FALSE;
	unsigned int max_cores = 4;
	nplist *nptimesT = NULL, *nptimesU = NULL, *npweightsT = NULL, *npweightsU = NULL, *npnanosT = NULL, *npnanosU = NULL;
	Bins *bins = NULL;
	Edges *edges = NULL;
	FuncCode func_code = FC_ERROR;
	// pointers to final data output
	cc_times *corrI = NULL;
	cc_weights *corrl = NULL;
	// variables for specifying the size and shape of the output array
	int outndim = 1;
	size_t nWt = 1, nWu=1, numel = 0;
	npy_intp *outdims = NULL;
	// parse arguments
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|OOOOOpppppI", kwlist, &pytimesT, &pytimesU, &pybins, &pyweightsT, &pyweightsU, &pynanosT, &pynanosU, &pyedges, &normalize, &norm_bin_width, &minzero, &cross_correlate, &validate, &max_cores))
		goto exit;
	if (max_cores == 0)
		max_cores = 1;
	// identify which type of correlation to perform
	if ((func_code = kwarg_choice(pyweightsT, pyweightsU, pynanosT, pynanosU)) == FC_ERROR)
		goto exit;
	// convert timesT/U
	if (base_casts(pytimesT, pytimesU, pybins, pyedges, minzero, validate, &nptimesT, &nptimesU, &bins, &edges))
		goto exit;
	// allocate array for normalization factor
	if (func_code == FC_WEIGHTSLIST)
	{
		if (weights_cast(pyweightsT, pyweightsU, nptimesT, nptimesU, &npweightsT, &npweightsU))
			goto frees;
		if ((outndim = get_weightslist_ndim(nptimesT->cdata.times, nptimesU->cdata.times, npweightsT->cdata.weightsL, npweightsU->cdata.weightsL, cross_correlate)) < 1)
			goto frees;
		nWt = npweightsT->cdata.weightsL->nW;
		nWu = npweightsU->cdata.weightsL->nW;
	}
	else if ( func_code == FC_WEIGHTSINDEX )
	{
		if (weights_index_cast(pyweightsT, pyweightsU, pynanosT, pynanosU, nptimesT, nptimesU, &npweightsT, &npweightsU, &npnanosT, &npnanosU))
			goto frees;
		if ((outndim = get_weightsnanos_ndim(nptimesT->cdata.times, nptimesU->cdata.times, npweightsT->cdata.weightsI, npweightsU->cdata.weightsI, 
																						npnanosT->cdata.nanos, npnanosU->cdata.nanos, cross_correlate)) < 1)
			goto frees;
		nWt = npweightsT->cdata.weightsI->nW;
		nWu = npweightsU->cdata.weightsI->nW;
	}
	if ((outdims = (npy_intp*) malloc(outndim*sizeof(npy_intp))) == NULL) {
		PyErr_NoMemory();
		goto frees;
	}
	outdims[outndim-1] = bins->nbin - 1;
	if (outndim == 2) {
		outdims[0] = (nWt != 1) ? (npy_intp) nWt : (npy_intp) nWu;
		numel = (bins->nbin - 1) * (size_t) outdims[0];
	}
	else if (outndim == 3) {
		outdims[0] = (npy_intp) nWt;
		outdims[1] = (npy_intp) nWu;
		numel = (bins->nbin - 1)*nWt*nWu;
	}
	else {
		numel = bins->nbin - 1;
	}
	// allocating for histogramed output
	if ((func_code == FC_INT)&&(!normalize)&&(!norm_bin_width)) {// if no normalization an no weights, then output is uint64, otherwise double
		if ((out = PyArray_SimpleNew(outndim, outdims, NPY_UINT64)) == NULL) {
			PyErr_NoMemory();
			goto frees;
		}
		corrI = (cc_times*) PyArray_DATA((PyArrayObject*) out);
		for (size_t i = 0; i < (bins->nbin - 1); i++)
			corrI[i] = 0;
	}
	else
	{
		if ((out = PyArray_SimpleNew(outndim, outdims, NPY_DOUBLE)) == NULL) {
			PyErr_NoMemory();
			goto frees;
		}
		corrl = (cc_weights*) PyArray_DATA((PyArrayObject*) out);
		for (size_t i = 0; i < numel; i++)
			corrl[i] = 0.0;
	}
	Py_BEGIN_ALLOW_THREADS;
	switch (func_code)
	{
		case FC_INT:
			if (normalize||norm_bin_width)
			{
				cerr = interface_correlate_int(nptimesT->cdata.times->n, edges->edges, nptimesT->cdata.times->lens, nptimesT->cdata.times->times,
										nptimesU->cdata.times->lens, nptimesU->cdata.times->times, bins->nbin, bins->bins, corrl, max_cores, normalize, norm_bin_width);
			}
			else
			{
				cerr = interface_correlate_int_hist(nptimesT->cdata.times->n, edges->edges, nptimesT->cdata.times->lens, nptimesT->cdata.times->times,
													nptimesU->cdata.times->lens, nptimesU->cdata.times->times, bins->nbin, bins->bins, corrI, max_cores);
			}
			break;
		case FC_WEIGHTSLIST:
			cerr = interface_correlate_weight(nptimesT->cdata.times->n, edges->edges, 
												nptimesT->cdata.times->lens, npweightsT->cdata.weightsL->nW, nptimesT->cdata.times->times, npweightsT->cdata.weightsL->weights,
												nptimesU->cdata.times->lens, npweightsU->cdata.weightsL->nW, nptimesU->cdata.times->times, npweightsU->cdata.weightsL->weights,
												bins->nbin, bins->bins, corrl, max_cores, normalize, norm_bin_width, cross_correlate);
			break;
		case FC_WEIGHTSINDEX:
			cerr = interface_correlate_weight_index(nptimesT->cdata.times->n, edges->edges,
					nptimesT->cdata.times->lens, npweightsT->cdata.weightsI->nW, nptimesT->cdata.times->times, npweightsT->cdata.weightsI->weights, npnanosT->cdata.nanos->nanos,
					nptimesU->cdata.times->lens, npweightsU->cdata.weightsI->nW, nptimesU->cdata.times->times, npweightsU->cdata.weightsI->weights, npnanosU->cdata.nanos->nanos,
					bins->nbin, bins->bins, corrl, max_cores, normalize, norm_bin_width, cross_correlate);
			break;
		case FC_ERROR:
			PyErr_SetString(PyExc_RuntimeError, "Error in C code, trying to correlate with unknown inputs");
			break;
	}
	Py_END_ALLOW_THREADS;
	if (cerr)
	{
		Py_XDECREF(out);
		PyErr_NoMemory();
		goto frees;
	}
	frees:
	Xfree(outdims);
	outdims = NULL;
	all_frees(nptimesT, nptimesU, bins, edges, npweightsT, npweightsU, npnanosT, npnanosU);
	npnanosU = NULL;
	npnanosT = NULL;
	npweightsU = NULL;
	npweightsT = NULL;
	edges = NULL;
	bins = NULL;
	nptimesU = NULL;
	nptimesT = NULL;
	exit:
	return out;
}

	
static PyObject* pyFCS_normalization_factor(PyObject *self, PyObject *args, PyObject* kwargs)
{
	char *kwlist[] = {"timesT", "timesU", "bins", "weightsT", "weightsU", "nanosT", "nanosU", "edges", "normalize", "norm_bin_width", "minzero", "cross_correlate", "validate", NULL};
	PyObject *pytimesT = NULL, *pytimesU = NULL, *pybins = NULL, *pyweightsT = NULL, *pyweightsU = NULL, *pynanosT = NULL, *pynanosU = NULL, *pyedges = NULL;
	int normalize = TRUE, norm_bin_width = TRUE, minzero=FALSE, cross_correlate = FALSE, validate = TRUE, cerr = FALSE;
	nplist *nptimesT = NULL, *nptimesU = NULL, *npweightsT = NULL, *npweightsU = NULL, *npnanosT = NULL, *npnanosU = NULL;
	Bins *bins = NULL;
	Edges *edges = NULL;
	cc_weights *norm = NULL;
	FuncCode func_code = FC_ERROR;
	size_t nWt = 1, nWu = 1, numel = 0;
	int outndim = 1;
	npy_intp *outdims = NULL;
	PyObject *out = NULL;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|OOOOOppppp", kwlist, &pytimesT, &pytimesU, &pybins, &pyweightsT, &pyweightsU, &pynanosT, &pynanosU, &pyedges, &normalize, &norm_bin_width, &minzero, &cross_correlate, &validate))
		goto exit;
	if ((func_code = kwarg_choice(pyweightsT, pyweightsU, pynanosT, pynanosU)) == FC_ERROR)
		goto exit;
	// convert timesT/U
	if (base_casts(pytimesT, pytimesU, pybins, pyedges, minzero, validate, &nptimesT, &nptimesU, &bins, &edges))
		goto exit;
	// allocate array for normalization factor
	if (func_code == FC_WEIGHTSLIST)
	{
		if (weights_cast(pyweightsT, pyweightsU, nptimesT, nptimesU, &npweightsT, &npweightsU))
			goto frees;
		if ((outndim = get_weightslist_ndim(nptimesT->cdata.times, nptimesU->cdata.times, npweightsT->cdata.weightsL, npweightsU->cdata.weightsL, cross_correlate)) < 1)
			goto frees;
		nWt = npweightsT->cdata.weightsL->nW;
		nWu = npweightsU->cdata.weightsL->nW;
	}
	else if ( func_code == FC_WEIGHTSINDEX )
	{
		if (weights_index_cast(pyweightsT, pyweightsU, pynanosT, pynanosU, nptimesT, nptimesU, &npweightsT, &npweightsU, &npnanosT, &npnanosU))
			goto frees;
		if ((outndim = get_weightsnanos_ndim(nptimesT->cdata.times, nptimesU->cdata.times, npweightsT->cdata.weightsI, npweightsU->cdata.weightsI, 
																						npnanosT->cdata.nanos, npnanosU->cdata.nanos, cross_correlate)) < 1)
			goto frees;
		nWt = npweightsT->cdata.weightsI->nW;
		nWu = npweightsU->cdata.weightsI->nW;
	}
	if ((outdims = (npy_intp*) malloc(outndim*sizeof(npy_intp))) == NULL) {
		PyErr_NoMemory();
		goto frees;
	}
	outdims[outndim-1] = bins->nbin - 1;
	if (outndim == 2) {
		outdims[0] = (nWt != 1) ? (npy_intp) nWt : (npy_intp) nWu;
		numel = (bins->nbin - 1) * (size_t) outdims[0];
	}
	else if (outndim == 3) {
		outdims[0] = (npy_intp) nWt;
		outdims[1] = (npy_intp) nWu;
		numel = (bins->nbin - 1)*nWt*nWu;
	}
	else {
		numel = bins->nbin - 1;
	}
	if ((out = PyArray_SimpleNew(outndim, outdims, NPY_DOUBLE)) == NULL) {
		PyErr_NoMemory();
		goto frees;
	}
	norm = (cc_weights*) PyArray_DATA((PyArrayObject*) out);
	for (size_t i = 0; i < numel; i++)
		norm[i] = 1.0;
	if ((!normalize)&&(!norm_bin_width)) {
		PyErr_WarnEx(PyExc_UserWarning, "No normalization factor calculated, at least one of normalize and norm_bin_width should be true", 1);
	}
	switch (func_code)
	{
		case FC_INT:
			cerr = interface_normalization_factor(nptimesT->cdata.times->n, edges->edges, nptimesT->cdata.times->lens, nptimesT->cdata.times->times, 
																						nptimesU->cdata.times->lens, nptimesU->cdata.times->times,
																						bins->nbin, bins->bins, norm, normalize, norm_bin_width);
			break;
		case FC_WEIGHTSLIST:
			cerr = interface_normalization_factor_weight(nptimesT->cdata.times->n, edges->edges, 
						nptimesT->cdata.times->lens, npweightsT->cdata.weightsL->nW, nptimesT->cdata.times->times, npweightsT->cdata.weightsL->weights,
						nptimesU->cdata.times->lens, npweightsU->cdata.weightsL->nW, nptimesU->cdata.times->times, npweightsU->cdata.weightsL->weights,
						bins->nbin, bins->bins, norm, normalize, norm_bin_width, cross_correlate);
			break;
		case FC_WEIGHTSINDEX:
			cerr = interface_normalization_factor_weight_index(nptimesT->cdata.times->n, edges->edges, 
						nptimesT->cdata.times->lens, npweightsT->cdata.weightsI->nW, nptimesT->cdata.times->times, npweightsT->cdata.weightsI->weights, npnanosT->cdata.nanos->nanos,
						nptimesU->cdata.times->lens, npweightsU->cdata.weightsI->nW, nptimesU->cdata.times->times, npweightsU->cdata.weightsI->weights, npnanosU->cdata.nanos->nanos,
						bins->nbin, bins->bins, norm, normalize, norm_bin_width, cross_correlate);
			break;
		case FC_ERROR:
			PyErr_SetString(PyExc_RuntimeError, "Error in C code, trying to compute normalization factor with uknown inputs");
			break;
	}
	if (cerr)
	{
		PyErr_NoMemory();
		goto frees;
	}
	frees:
	Xfree(outdims);
	outdims = NULL;
	all_frees(nptimesT, nptimesU, bins, edges, npweightsT, npweightsU, npnanosT, npnanosU);
	npnanosU = NULL;
	npnanosT = NULL;
	npweightsU = NULL;
	npweightsT = NULL;
	edges = NULL;
	bins = NULL;
	nptimesU = NULL;
	nptimesT = NULL;
	exit:
	return out;
}

static PyObject* pyFCS_normalize(PyObject *self, PyObject *args, PyObject *kwargs)
{
	char *kwlist[] = {"G", "timesT", "timesU", "bins", "weightsT", "weightsU", "nanosT", "nanosU", "edges", "normalize", "norm_bin_width", "minzero", "validate", NULL};
	PyObject *pycorrl = NULL, *pytimesT = NULL, *pytimesU = NULL, *pybins = NULL, *pyweightsT = NULL, *pyweightsU = NULL, *pynanosT = NULL, *pynanosU = NULL, *pyedges = NULL;
	nplist *nptimesT = NULL, *nptimesU = NULL, *npweightsT = NULL, *npweightsU = NULL, *npnanosT = NULL, *npnanosU = NULL;
	PyArrayObject *npcorrl = NULL;
	Bins *bins = NULL;
	Edges *edges = NULL;
	cc_weights *corrl = NULL;
	FuncCode func_code = FC_ERROR;
	int ndim = 0;
	int normalize = TRUE, norm_bin_width = TRUE, minzero=FALSE, validate = TRUE;
	PyObject *out = NULL;
	int cerr = FALSE, cross_correlate = -1;
	npy_intp lbin = 0, nWt = 1, nWu = 1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|OOOOOpppp", kwlist, &pycorrl, &pytimesT, &pytimesU, &pybins, &pyweightsT, &pyweightsU, &pynanosT, &pynanosU, &pyedges, &normalize, &norm_bin_width, &minzero, &validate))
		goto final;
	if ((!normalize)&&(!norm_bin_width))
		PyErr_WarnEx(PyExc_UserWarning, "No normalization factor calculated, at least one of normalize and norm_bin_width should be true", 1);
	if ((func_code = kwarg_choice(pyweightsT, pyweightsU, pynanosT, pynanosU)) == FC_ERROR)
		goto final;
	if (base_casts(pytimesT, pytimesU, pybins, pyedges, minzero, validate, &nptimesT, &nptimesU, &bins, &edges))
		goto frees;
	if ((npcorrl = npArray_fromany(pycorrl, NPY_DOUBLE, 1, 3, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_ENSURECOPY)) == NULL)
	{
		PyErr_SetString(PyExc_TypeError, "correlation must be 1, 2, or 3D array with last dimension 1 less than length of bins");
		goto frees;
	}
	ndim = PyArray_NDIM(npcorrl);
	if ((lbin = PyArray_DIM(npcorrl, ndim - 1)) != (npy_intp)(bins->nbin - 1))
	{
		PyErr_Format(PyExc_ValueError, "last dimension of G and bins do no match, with that of G one less than bins, got %ld and %ld respectively", lbin, bins->nbin); 
		goto frees;
	}
	// check matching weights dimensions etc.
	switch(func_code)
	{
		case FC_INT:
			if (ndim != 1)
			{
				PyErr_Format(PyExc_ValueError, "G (array to be normalized) must be 1D for standard correlation (no weights), but input G is %dD", ndim);
				cerr = TRUE;
			}
			break;
		case FC_WEIGHTSLIST:
			if ((cerr = weights_cast(pyweightsT, pyweightsU, nptimesT, nptimesU, &npweightsT, &npweightsU)))
				break;
			if ((cerr = check_weightslist(nptimesT->cdata.times, nptimesU->cdata.times, npweightsT->cdata.weightsL, npweightsU->cdata.weightsL)))
				break;
			nWt = (npy_intp) npweightsT->cdata.weightsL->nW;
			nWu = (npy_intp) npweightsU->cdata.weightsL->nW;
			break;
		case FC_WEIGHTSINDEX:
			if ((cerr = weights_index_cast(pyweightsT, pyweightsU, pynanosT, pynanosU, nptimesT, nptimesU, &npweightsT, &npweightsU, &npnanosT, &npnanosU)))
				break;
			if ((cerr = check_weightsnanos(nptimesT->cdata.times, nptimesU->cdata.times, npweightsT->cdata.weightsI, npweightsU->cdata.weightsI, npnanosT->cdata.nanos, npnanosU->cdata.nanos)))
				break;
			nWt = (npy_intp) npweightsT->cdata.weightsI->nW;
			nWu = (npy_intp) npweightsU->cdata.weightsI->nW;
			break;
		case FC_ERROR:
			cerr = TRUE;
			PyErr_SetString(PyExc_RuntimeError, "Error in C code, trying to normalize with unexpected inputs");
			break;
	}
	if (cerr)
		goto frees;
	// check that shapes of G matches weights etc...
	if (func_code != FC_INT)
	{
		if (ndim == 1)
		{
			if ((nWt != 1)||(nWu != 1))
			{
				cerr = TRUE;
				PyErr_SetString(PyExc_ValueError, "For 1D G, weightsT/U must both have a single weights array (1D)");
				goto frees;
			}
			cross_correlate = FALSE;
		}
		else if (ndim == 2)
		{
			npy_intp nW = PyArray_DIM(npcorrl, 0);
			if ((nWt != nW)&&(nWu != nW)) // at least 1 must be nW, both means error
			{
				PyErr_Format(PyExc_ValueError, "If G is 2D, one or both weightsT/U must have same number of weights functions as specified in the dimension 0 of G, got %ld for G, %ld for weightsT, and %ld for weightsU", nW, nWt, nWu);
				cerr = TRUE;
				goto frees;
			}
			if (nWt != nWu) // cross correlate because nWt aren nWu are not equal, (next if ensures has dimension of 1)
			{
				if ((nWt != 1) && (nWu != 1)) // since nWt != nWu, one of them must be 1
				{
					PyErr_Format(PyExc_ValueError, "If G is 2D, weightsT/U must either have the same number of weights functions, or one of them have a single weights function, got %ld, %ld weights functions for weightsT and weightsU respectively", nWt, nWu);
					cerr = TRUE;
					goto frees;
				}	
				cross_correlate = TRUE;
			}
			else
				cross_correlate = FALSE; 
		}
		else if (ndim == 3)
		{
			cross_correlate = TRUE;
			if ((nWt != PyArray_DIM(npcorrl, 0)) || (nWu != PyArray_DIM(npcorrl, 1)))
			{
				PyErr_Format(PyExc_ValueError, "If G is 3D, weightsT/U dimension 0 must match G dimentions 0 and 1 of G respectively got, %ld for weightsT and %ld for weightsU, expecting %ld and %ld respectively", nWt, nWu, PyArray_DIM(npcorrl, 0), PyArray_DIM(npcorrl, 1));
				cerr = TRUE;
				goto frees;
			}
		}
	}
	corrl = (cc_weights*) PyArray_DATA(npcorrl);
	switch(func_code)
	{
		case FC_INT:
			if ((cerr = interface_normalize(nptimesT->cdata.times->n, edges->edges, nptimesT->cdata.times->lens, nptimesT->cdata.times->times,
																			nptimesU->cdata.times->lens, nptimesU->cdata.times->times,
																			bins->nbin, bins->bins, corrl, normalize, norm_bin_width)))
				PyErr_NoMemory();
			break;
		case FC_WEIGHTSLIST:
			if ((cerr = interface_normalize_weight(nptimesT->cdata.times->n, edges->edges,
												nptimesT->cdata.times->lens, npweightsT->cdata.weightsL->nW, nptimesT->cdata.times->times, npweightsT->cdata.weightsL->weights,
												nptimesU->cdata.times->lens, npweightsU->cdata.weightsL->nW, nptimesU->cdata.times->times, npweightsU->cdata.weightsL->weights,
												bins->nbin, bins->bins, corrl, normalize, norm_bin_width, cross_correlate)))
				PyErr_NoMemory();
			break;
		case FC_WEIGHTSINDEX:
			if ((cerr = interface_normalize_weight_index(nptimesT->cdata.times->n, edges->edges,
												nptimesT->cdata.times->lens, npweightsT->cdata.weightsI->nW, nptimesT->cdata.times->times, npweightsT->cdata.weightsI->weights, npnanosT->cdata.nanos->nanos,
												nptimesU->cdata.times->lens, npweightsU->cdata.weightsI->nW, nptimesU->cdata.times->times, npweightsU->cdata.weightsI->weights, npnanosU->cdata.nanos->nanos,
												bins->nbin, bins->bins, corrl, normalize, norm_bin_width, cross_correlate)))
				PyErr_NoMemory();
			break;
		case FC_ERROR:
			PyErr_SetString(PyExc_RuntimeError, "Error in C code, misassigned inputs");
			cerr = TRUE;
			break;
	}
	frees:
	if (cerr)
	{
		Py_XDECREF(npcorrl);
		npcorrl = NULL;
	}
	if (npcorrl != NULL)
		out = (PyObject*) npcorrl;
	all_frees(nptimesT, nptimesU, bins, edges, npweightsT, npweightsU, npnanosT, npnanosU);
	npnanosU = NULL;
	npnanosT = NULL;
	npweightsU = NULL;
	npweightsT = NULL;
	nptimesT = NULL;
	nptimesU = NULL;
	bins = NULL;
	edges = NULL;
	final:
	return out;
}

static PyMethodDef pyFCS_funcs[] = {
	{"correlate", (PyCFunction)pyFCS_correlate, METH_VARARGS|METH_KEYWORDS, 
		"correlate(timesT, timesU, bins, weightsT=None, weightsU=None, nanosT=None, nanosU=None, edges=None, normalize=True, norm_bin_width=True, minzero=False, cross_correlate=False, validate=True, max_cores=4)\n"
		"--\n\n"
		"Compute the correlation between two arrays, or sets of arrays,\n" 
		"of discrete events (point-process). The input values should be\n"
		"the result of a point process such as photon arival times.\n"
		"This is used to compute the correlation curves used in\n"
		"Fluorescence Coorelation Spectroscopy (FCS), and extensions thereof.\n"
		"The main extensions supported are: Fluorescence Lifetime Correlation\n"
		"Spectroscopy (FLCS, `(Bohmer 2002) <https://doi.org/10.1016/S0009-2614(02)00044-1>`_ ),\n"
		"and purified/filtered FCS\n"
		"`(Laurence 2007) <https://doi.org/10.1529/biophysj.106.093591>`_\n"
		"/`(Felekyan 2012) <https://doi.org/10.1002/cphc.201100897>`_\n"
		"\n"
		"The basic form of this function implements the algorithm described in\n"
		"`(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`_ .\n"
		"\n"
		"In this function, the (normalized) point-process correlation is defined as\n"
		"\n"
		".. math::\n"
		"    \\hat{G}(\\tau_{b}) = \\frac{\\sum_{K=1}^{K}{n(\\{(i, j) \\ni \\tau_{b} \\le u_{k,j} - t_{k,i} < \\tau_{b+1}\\})}\n"
		"    (T_{k}-\\tau_{b+1})}\n"
		"    {\\sum_{m=1}^{M}{n(\\{i \\ni t_{k,i} \\le T - \\tau_{b+1}\\})}\n"
		"    \\sum_{m=1}^{M}{n(\\{j \\ni u_{k,j} \\ge \\tau_{b+1}\\})}}\n"
		"\n"
		"\n"
		"Where the function :math:`n` is the counts that satisfy the condition,\n"
		":math:`T` and :math`U` are arrays of arrays of photons, whose elements are\n"
		"denoted :math:`t_{ki}` and :math:`u_{kj}` respectively. Both\n"
		":math:`T` and :math:`K` have :math:`K` distinct arrays. The\n"
		"number of elements within array are independent of each other\n"
		"\n"
		"The core algorithm from Laurence 2006 divides this computation into\n"
		"Two key algorithms, the first computes\n"
		"\n"
		".. math::\n"
		"    \\hat{C}(\\tau_{b}) = \\sum_{k=1}^{K}{n(\\{(i, j) \\ni \\tau_{b} \\le u_{k,j} - t_{k,i} < \\tau_{b+1})\\})}\n"
		"\n"
		"\n"
		"and the second\n"
		"\n"
		".. math::\n\n"
		"    \\frac{(T_{k}-\\tau_{b+1})}\n"
		"    {\\sum_{m=1}^{M}{n(\\{i \\ni t_{k,i} \\le T - \\tau_{b+1}\\})}\n"
		"    \\sum_{m=1}^{M}{n(\\{j \\ni u_{k,j} \\ge \\tau_{b+1}\\})(\\tau_{b+1}-\\tau_{b})}}\n\n"
		"Additionally, since the alorithm groups values of :math:`\\tau` into bins,\n"
		"the raw counts are divided by the width of the bin.\n"
		"\n"
		"For maximal flexibility, this functions has keyword arguments allowing\n"
		"these two normalizations to be ignored, ``normalize=False`` will\n"
		"cause the returned values **not** be multiplied by\n"
		":math:`\\frac{\\sum_{k}{T_{k}-\\tau}}{\\sum_{k}{n(\\{i \\ni t_{ki} \\le T_{k} - \\tau\\})}\\sum_{k}{n(\\{j \\ni u_{kj} \\ge \\tau\\})}}`.\n"
		"While ``norm_bin_width=False`` causes the values **not** to be\n"
		"divided by the width of the given :math:`\\tau`.\n"
		"\n"
		"Weights can also be incorporated. This is used in FLCS, (see\n"
		"`Ghosh 2018 <https://doi.org/10.1016/j.ymeth.2018.02.009>`_ for\n"
		"review). When weights are incorporated, the value of each photon\n"
		"pair is the product of the weight assigned to each photon, resulting\n"
		"in the following modified formula:\n\n"
		".. math::\n\n"
		"    \\hat{G}_{\\alpha,\\beta}(\\tau_{b}) = \\frac{\\{\\sum_{m=1}^{M}{[\\sum_{i,j}^{\\tau_{b} \\le t_{j}-t_{i} < \\tau_{b+1}}{f_{\\alpha,i}f_{\\beta,j}}]}\\}\n"
		"    \\sum_{m=1}^{M}{(T-\\tau_{b})}}\n"
		"    {\\sum_{m=1}^{M}{[\\sum_{i}^{t_{i} \\le T - \\tau_{b}}}{f_{\\alpha,i}]}\n"
		"    \\sum_{m=1}^{M}{[\\sum_{j}^{t_{j} \ge \\tau_{b}}{f_{\\beta,j}}}]}\n\n"
		"Where :math:`\\delta` is the delta function, and :math:`H` is the\n"
		"heavyside function. The effect is simply to multiply each photon\n"
		"or photon pair by its respective weight(s).\n\n"
		"Parameters\n"
		"----------\n"
		"timesT/U: list[numpy.ndarray] | numpy.ndarray\n"
		"    Arrival times of points (photons) to cross-correlate.\n"
		"    Must be non-negative integer array, or a sequence of non-negative integer arrays\n"
		"bins: numpy.ndarray\n"
		"    Bin edges for values of :math:`\tau`\n"
		"weightsT/U: list[numpy.ndarray] | numpy.ndarray, optional\n"
		"    Array(s) of weights for points( photons). The shape of the arrays(s)\n"
		"    is determined by whether or not nanosT/U are also specified\n"
		"    if nanosT/U are specified, then weightsT/U is treated as a\n"
		"    look-up table with which nanosT/U indexes into. In this case\n"
		"    weightsT/U is a 1 or 2D array (specifying a single or multiple\n"
		"    weights functions respectively), and the last dimension of\n"
		"    weightsT/U must be large enough to be indexed by nanosT/U.\n"
		"    If nanosT/U are not specified, then weightsT/U must specify\n"
		"    weights per point (photon) in timesT/U. Multiple weights functions\n"
		"    may be specified by making the array(s) 2D, and having\n"
		"    dimension 0 specify the distinct functions\n"
		"    The default is None\n"
		"nanosT/U: list[numpy.ndarray] | numpy.ndarray, optional\n"
		"    Indexes per point (photon) to identify which weight to use.\n"
		"    Must be non-negative integer array or sequence of non-negative interger arrays\n"
		"    with size(s) matching that of timesT/U. The default is None.\n"
		"edges: numpy.ndarray, optional\n"
		"    1 or 2D array, defining the start/stop times of each burst.\n"
		"    This is used exclusively in computation of normalization factors.\n"
		"    The last dimension must have 2 elements (i.e. start and stop).\n"
		"    If 1D but multiple arrays are specified in timesT/U, then\n"
		"    the same start stop times are used for all bursts. If not\n"
		"    specified, then edges are assigned automatically as first and\n"
		"    last times in timesT/U arrays, unless ``minzero=TRUE``, where\n"
		"    the start time is assumed to be 0, while the stop remains the\n"
		"    last photon. The default is None.\n"
		"normalize: bool, optional\n"
		"    Whether or not to multiply array by normalization factor.\n"
		"    The default is True.\n"
		"norm_bin_width: bool, optional\n"
		"    Whether or not to divide by width of bins. The default is True.\n"
		"minzero: bool, optional\n"
		"    When using automatic assement of start/stop times for normalization\n"
		"    True will make the start time always 0, False (default),\n"
		"    then use the first photon in the times array(s) for normalization\n"
		"cross_correlate: bool, optional\n"
		"    Whether or not to return the correlations of all combinations\n"
		"    of weightsT with weightsU arrays (True), or only correlate each\n"
		"    weight with it's pair in weightsT/U (False).\n"
		"    *For weights correlation only, ignored if only timesT/U specified.*\n"
		"    the default is False.\n"
		"    Whether or not to performa a check that times arrays are\n"
		"    monotonically increasing, if True (default), will raise an\n"
		"    error if photons are not monotinically increasing. If false\n"
		"    no error will be raised, but if photons are no monotonically\n"
		"    increasing, the results will be non-defined. The default is True.\n"
		"validate: bool, optional\n"
		"    Whether or not to perform a check on timesT/U to ensure that\n"
		"    both are monotonically increasing. The algorithm assumes\n"
		"    arrays are monotonically increasing, and non-monotonically\n"
		"    increasing arrays will have undefined results.\n"
		"    The default is True.\n"
		"max_cores: int, optional\n"
		"    Maximum number of cores to use in computation, for parallel \n"
		"    processing optimization (only applicable for multiple bursts).\n"
		"    The default is 4.\n"
		"Returns\n"
		"-------\n"
		"G: numpy.ndarray\n"
		"    Correlation array, representing the correlation between the\n"
		"    T/U arrays, with the normalizations specified in ``normalize``\n"
		"    and ``norm_bin_width``. If the correlations use weighting\n"
		"    functions, then the dimensionality will be determined by the\n"
		"    input to ``cross_correlate``, resulting in 2D \n"
		"    when ``cross_correlate=False``, and 3D when\n"
		"    ``cross_correlate=True``.\n"
	},
	{"normalization_factor",(PyCFunction)pyFCS_normalization_factor, METH_VARARGS|METH_KEYWORDS, 
		"normalization_factor(timesT, timesU, bins, weightsT=None, weightsU=None, nanosT=None, nanosU=None, edges=None, normalize=True, norm_bin_width=True, minzero=False, validate=True)\n"
		"--\n\n"
		"Compute the normalization factor (see :func:`correlate` for a set of\n"
		"times and bins.\n\n"
		"\n"
		"Parameters\n"
		"----------\n"
		"timesT/U: list[numpy.ndarray] | numpy.ndarray\n"
		"    Arrival times of points (photons) to cross-correlate.\n"
		"    Must be non-negative integer array, or a sequence of non-negative integer arrays\n"
		"bins: numpy.ndarray\n"
		"    Bin edges for values of :math:`\tau`\n"
		"weightsT/U: list[numpy.ndarray] | numpy.ndarray, optional\n"
		"    Array(s) of weights for points( photons). The shape of the arrays(s)\n"
		"    is determined by whether or not nanosT/U are also specified\n"
		"    if nanosT/U are specified, then weightsT/U is treated as a\n"
		"    look-up table with which nanosT/U indexes into. In this case\n"
		"    weightsT/U is a 1 or 2D array (specifying a single or multiple\n"
		"    weights functions respectively), and the last dimension of\n"
		"    weightsT/U must be large enough to be indexed by nanosT/U.\n"
		"    If nanosT/U are not specified, then weightsT/U must specify\n"
		"    weights per point (photon) in timesT/U. Multiple weights functions\n"
		"    may be specified by making the array(s) 2D, and having\n"
		"    dimension 0 specify the distinct functions\n"
		"    The default is None\n"
		"nanosT/U: list[numpy.ndarray] | numpy.ndarray, optional\n"
		"    Indexes per point (photon) to identify which weight to use.\n"
		"    Must be non-negative integer array or sequence of non-negative interger arrays\n"
		"    with size(s) matching that of timesT/U. The default is None.\n"
		"edges: numpy.ndarray, optional\n"
		"    1 or 2D array, defining the start/stop times of each burst.\n"
		"    The last dimension must have 2 elements (i.e. start and stop).\n"
		"    If 1D but multiple arrays are specified in timesT/U, then\n"
		"    the same start stop times are used for all bursts. If not\n"
		"    specified, then edges are assigned automatically as first and\n"
		"    last times in timesT/U arrays, unless ``minzero=TRUE``, where\n"
		"    the start time is assumed to be 0, while the stop remains the\n"
		"    last photon. The default is None.\n"
		"normalize: bool, optional\n"
		"    Whether or not to multiply array by normalization factor.\n"
		"    The default is True.\n"
		"norm_bin_width: bool, optional\n"
		"    Whether or not to divide by width of bins. The default is True.\n"
		"minzero: bool, optional\n"
		"    When using automatic assement of start/stop times for normalization\n"
		"    True will make the start time always 0, False (default),\n"
		"    then use the first photon in the times array(s) for normalization\n"
		"cross_correlate: bool, optional\n"
		"    Whether or not to return the normalization factor for all combinations\n"
		"    of weightsT with weightsU arrays (True), or only correlate each\n"
		"    weight with it's pair in weightsT/U (False).\n"
		"    *For weights correlation only, ignored if only timesT/U specified.*\n"
		"    the default is False.\n"
		"validate: bool, optional\n"
		"    Whether or not to perform a check on timesT/U to ensure that\n"
		"    both are monotonically increasing. The algorithm assumes\n"
		"    arrays are monotonically increasing, and non-monotonically\n"
		"    increasing arrays will have undefined results.\n"
		"    The default is True.\n"
		"Returns\n"
		"-------\n"
		"norm_factor: np.ndarray\n"
		"    The normalization factors for each bin of the array.\n"
		"    Multiply this by the calculated non-normalzied correlation\n"
		"    for the normalized curve.\n"
	},
	{"normalize", (PyCFunction)pyFCS_normalize, METH_VARARGS|METH_KEYWORDS,
		"normalize(G, timesT, timesU, bins, edges=None, normalize=True, norm_bin_width=True, minzero=False, validate=True)\n"
		"--\n\n"
		"Take input of non-normalized correlation and apply specified\n"
		"normalization factors.\n\n"
		".. note::\n\n"
		"    The values for `normalize` and `norm_bin_width` should be \n"
		"    converse of those used in the :func:`correlate`\n\n"
		"Normalize (normalize keyword argument) point-process cross-correlation function.\n\n"
		"This normalization is usually employed for fluorescence correlation\n"
		"spectroscopy (FCS) analysis.\n"
		"The normalization is performed according to\n"
		"`(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`_ .\n"
		"Basically, the input argument `G` is multiplied by:\n\n"
		".. math::\n"
		"    \\frac{T-\\tau}{n(\\{i \\ni t_i \\le T - \\tau\\})n(\\{j \\ni u_j \\ge \\tau\\})}\n\n"
		"where `n({})` is the operator counting the elements in a set, *t* and *u*\n"
		"are the input arrays of the correlation, *τ* is the time lag and *T*\n"
		"is the measurement duration.\n\n"
		"Parameters\n"
		"----------\n"
		"G: numpy.ndarray\n"
		"    Raw cross-correlation to be normalized. May be either 1, 3 or\n"
		"    3D. Should be the ouptput of the :func:`correlate` function.\n"
		"    using identical times/weigths/nanosT/U and bin arguments.\n"
		"timesT/U: list[numpy.ndarray] | numpy.ndarray\n"
		"    Arrival times of points (photons) to cross-correlate.\n"
		"    Must be non-negative integer array, or a sequence of non-negative integer arrays\n"
		"bins: numpy.ndarray\n"
		"    Bin edges for values of :math:`\tau`\n"
		"weightsT/U: list[numpy.ndarray] | numpy.ndarray, optional\n"
		"    Array(s) of weights for points( photons). The shape of the arrays(s)\n"
		"    is determined by whether or not nanosT/U are also specified\n"
		"    if nanosT/U are specified, then weightsT/U is treated as a\n"
		"    look-up table with which nanosT/U indexes into. In this case\n"
		"    weightsT/U is a 1 or 2D array (specifying a single or multiple\n"
		"    weights functions respectively), and the last dimension of\n"
		"    weightsT/U must be large enough to be indexed by nanosT/U.\n"
		"    If nanosT/U are not specified, then weightsT/U must specify\n"
		"    weights per point (photon) in timesT/U. Multiple weights functions\n"
		"    may be specified by making the array(s) 2D, and having\n"
		"    dimension 0 specify the distinct functions\n"
		"    The default is None.\n"
		"nanosT/U: list[numpy.ndarray] | numpy.ndarray, optional\n"
		"    Indexes per point (photon) to identify which weight to use.\n"
		"    Must be non-negative integer array or sequence of non-negative interger arrays\n"
		"    with size(s) matching that of timesT/U. The default is None.\n"
		"edges: numpy.ndarray, optional\n"
		"    1 or 2D array, defining the start/stop times of each burst.\n"
		"    The last dimension must have 2 elements (i.e. start and stop).\n"
		"    If 1D but multiple arrays are specified in timesT/U, then\n"
		"    the same start stop times are used for all bursts. If not\n"
		"    specified, then edges are assigned automatically as first and\n"
		"    last times in timesT/U arrays, unless ``minzero=TRUE``, where\n"
		"    the start time is assumed to be 0, while the stop remains the\n"
		"    last photon. The default is None.\n"
		"normalize: bool, optional\n"
		"    Whether or not to multiply array by normalization factor.\n"
		"    Should be oposite of argument used in :func:`correlate`.\n"
		"    The default is True.\n"
		"norm_bin_width: bool, optional\n"
		"    Whether or not to divide by width of bins. Shoudl be oposite\n"
		"    of argument used in :func:`correlate`. The default is True.\n"
		"minzero: bool, optional\n"
		"    When using automatic assement of start/stop times for normalization\n"
		"    True will make the start time always 0, False (default),\n"
		"    then use the first photon in the times array(s) for normalization\n"
		"Returns\n"
		"-------\n"
		"G: numpy.ndarray\n"
		"    Array of normalized values for the cross-correlation function,\n"
		"    same size as the input argument `G`.\n"
		},
	{NULL,NULL,0,NULL}
};

static struct PyModuleDef pyFCS_module =
{
	PyModuleDef_HEAD_INIT, "pyFCS", 
	"Module for performing point process correlations. Primarily used in FCS and related methods.\n", -1,
	pyFCS_funcs
};	

PyMODINIT_FUNC PyInit_pyFCS(void)
{
	PyObject *module = PyModule_Create(&pyFCS_module);
	import_array();
	return module;
};
