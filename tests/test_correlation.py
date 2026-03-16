#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:36:44 2024

@author: paul
"""

import pytest
import numpy as np
from itertools import product, chain
from sys import getrefcount as grc

import pyFCS as fcs

def grcs(inp):
    out = (grc(inp), )
    if isinstance(inp, (list, tuple, np.ndarray)):
        out += tuple(grc(sub) for sub in inp if isinstance(sub, (list, tuple, np.ndarray)))
    return out

def ensure_refcount(func, args, kwargs, arg_exclude=None, kwarg_exclude=None):
    arg_exclude = tuple() if arg_exclude is None else arg_exclude
    kwarg_exclude = tuple(key for key, value in kwargs.items() if isinstance(value, (int, bool))) if kwarg_exclude is None else kwarg_exclude
    barg_count = tuple(grcs(arg) for i, arg in enumerate(args) if i not in arg_exclude)
    bkwarg_count = {key:grcs(value) for key, value in kwargs.items() if key not in kwarg_exclude}
    out = func(*args, **kwargs)
    oarg_count = tuple(grcs(arg) for i, arg in enumerate(args) if i not in arg_exclude)
    bkwarg_count = {key:grcs(value) for key, value in kwargs.items() if key not in kwarg_exclude}
    assert barg_count == oarg_count, "refcounts of args change after function call"
    assert all(value == bkwarg_count[key] for key, value in bkwarg_count), "refcounts of kwargs chnage after function call"
    return out

timesA = np.array([1,4,20,30,35,61,65,70,79,100,106,110,200,210,212,220,250,270], dtype=np.uint64)
timesB = np.array([2,5,18,22,33,60,64,68,72,74,78,105,192,198,204,214,218,250,260], dtype=np.uint64)

nanosA = np.array([1,8,5,3,7,10,9,6,2,0,4,1,4,1,7,6,0,7], dtype=np.uint16)
nanosB = np.array([2,5,1,8,3,6,4,8,7,9,0,10,9,1,4,2,8,5,6], dtype=np.uint16)

timesC = np.array(
    [
     np.array([500,523,584,785,788,801,813,840,1148,1150,1212,1238,1454,1582,1621], dtype=np.uint64),
     np.array([5171,5179,5782,5840,6710,6827,7056,7113,7134,7189,8650,8950,8978,9045], dtype=np.uint64),
     np.array([11358,11402,11418,11425,11432,11468,11689,11697,11721,11831,11917], dtype=np.uint64),
     np.array([23134,23145,23442,23488,23517,23521,23542,23610,23658,23672,23679,23695], dtype=np.uint64)
     ],
    dtype=object)

timesD = np.array(
    [
     np.array([520,528,564,574,788,798,811,815,832,845,1138,1161,1222,1231,1238,1458,1601,1611], dtype=np.uint64),
     np.array([5185,5792,5855,6818,7027,7193,7234,7250,8680,8910,8971,8995], dtype=np.uint64),
     np.array([11320,11388,11452,11468,11582,11599,11632,11668,11689,11697,11712,11751,11880,11957], dtype=np.uint64),
     np.array([23184,23198,23412,23509,23518,23528,23648,23668,23674,23698], dtype=np.uint64)
     ],
    dtype=object)

nanosC = np.array(
    [
     np.array([5,3,4,8,7,1,3,9,2,6,10,1,4,8,6], dtype=np.uint16),
     np.array([7,9,5,0,6,2,1,3,4,8,10,9,8,5], dtype=np.uint16),
     np.array([1,2,8,5,3,4,9,7,6,0,7], dtype=np.uint16),
     np.array([4,5,2,8,7,5,3,0,6,9,10,9], dtype=np.uint16)
     ],
    dtype=object)

nanosD = np.array(
    [
     np.array([5,8,4,7,0,9,1,6,2,3,1,10,2,3,8,5,6,10], dtype=np.uint16),
     np.array([5,2,8,6,7,3,4,0,1,9,7,10], dtype=np.uint16),
     np.array([0,8,2,6,5,9,1,3,4,7,10,5,10,9], dtype=np.uint16),
     np.array([4,8,2,9,3,5,6,10,0,1], dtype=np.uint16)
     ],
    dtype=object)

weights = np.array(
    [[ 0.1, 1.2, 0.5,-0.1,-1.8, 0.5, 0.8,-5.5,-1.5, 0.1, 0.1],
     [-0.1,-0.4,-1.3, 0.5, 2.5, 0.1, 0.9, 3.2,-0.4, 0.1,-0.2],
     [ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
    dtype=np.float64)

weightsconst = np.vstack([np.ones(11, dtype=np.float64), 
                          np.ones(11, dtype=np.float64)*2,
                          np.ones(11, dtype=np.float64)*3])


def make_weights_list(weight, nanos):
    if nanos.dtype == object:
        weights = np.empty(nanos.size, dtype=object)
        for i in range(nanos.size):
            weights[i] = weight[:,nanos[i]]
        return weights
    else:
        return weight[:,nanos]

weightsA = make_weights_list(weights, nanosA)
weightsAc = make_weights_list(weightsconst, nanosA)
weightsB = make_weights_list(weights, nanosB)
weightsBc = make_weights_list(weightsconst, nanosB)
weightsC = make_weights_list(weights, nanosC)
weightsCc = make_weights_list(weightsconst, nanosC)
weightsD = make_weights_list(weights, nanosD)
weightsDc = make_weights_list(weightsconst, nanosD)

bins = np.array([0, 10, 20, 40, 80], dtype=np.uint64)

def cast_int16(inp):
    return inp.astype(np.int16)

def cast_uint16(inp):
    return inp.astype(np.uint16)

def cast_int64(inp):
    return inp.astype(np.int64)

def cast_uint64(inp):
    return inp.astype(np.uint64)

def cast_float64(inp):
    return inp.astype(np.float64)

def cast_to(inp, tp):
    if inp.ndim == 1 and inp.dtype != object:
        to_pythontype = int if np.issubdtype(inp.dtype, np.integer) else float
        return tp(to_pythontype(i) for i in inp)
    return tp(cast_to(i, tp) for i in inp)


def cast_tuple(inp):
    return cast_to(inp, tuple)

def cast_list(inp):
    return cast_to(inp, list)

def outer_tuple(inp, inner):
    return tuple(inner(arr) for arr in inp)

def outer_list(inp, inner):
    return list(inner(arr) for arr in inp)

def outer_ndarray(inp, inner):
    out = np.empty(inp.shape, dtype=object)
    for i in range(inp.size):
        out[i] = inner(inp[i])
    return out


@pytest.fixture(scope='module', params=[cast_list, cast_tuple, cast_int64, cast_uint64])
def bins_types(request):
    return request.param(bins)

@pytest.fixture(scope='module', params=list((outer, inner) for i, (outer, inner) in
                                            enumerate(product([None, outer_list, outer_tuple],
                                                              [cast_list, cast_tuple, cast_int64, cast_uint64]))
                                            if i == 0 or outer is not None))
def times_types(request):
    outer, inner = request.param
    if outer is None:
        return inner(timesA), inner(timesB)
    return outer(timesC, inner), outer(timesD, inner)

@pytest.fixture(scope='module', params=[(timesA, timesA), (timesA, timesB), (timesC, timesC), (timesC, timesD)])
def times_base(request):
    return request.param

@pytest.fixture(scope='module', params=list((outer, inner) for i, (outer, inner) in
                                            enumerate(product([None, outer_list, outer_tuple],
                                                              [cast_list, cast_tuple, cast_float64]))
                                            if i == 0 or outer is not None))
def weights_types(request):
    outer, inner = request.param
    if outer is None:
        return timesA, timesB, inner(weightsA), inner(weightsB)
    return timesC, timesD, outer(weightsC, inner), outer(weightsD, inner)

@pytest.fixture(scope='module', params=list((outer, inner) for i, (outer, inner) in
                                            enumerate(product([None, outer_list, outer_tuple],
                                                              [cast_list, cast_tuple, cast_int16, cast_uint16]))
                                            if i == 0 or outer is not None))
def nanos_types(request):
    outer, inner = request.param
    if outer is None:
        return timesA, timesB, inner(nanosA), inner(nanosB)
    return timesC, timesD, outer(nanosC, inner), outer(nanosD, inner)


@pytest.fixture(scope='module', params=list())
def extra_kwargs(request):
    return {key:val for key, val in zip(('minzero', 'validate', 'max_cores')) if not isinstance(key, str) or key != 'None'}


def test_norms(times_base):
    tA, tB = times_base
    ref_tA, ref_tB = grc(tA), grc(tB)
    
    allc = fcs.correlate(tA, tB, bins, normalize=True, norm_bin_width=True)
    
    nilc = fcs.correlate(tA, tB, bins, normalize=False, norm_bin_width=False)
    niln = fcs.normalization_factor(tA, tB, bins, normalize=True, norm_bin_width=True)
    # assert np.allclose(niln*nilc, allc), "inconsistent correlation normalization with normalize=False, norm_bin_widht=False"
    nilG = fcs.normalize(nilc, tA, tB, bins, normalize=True, norm_bin_width=True)
    # assert np.allclose(nilG, allc), "inconsistent normalize with normalize=False, norm_bin_width=False"
    
    nrmc = fcs.correlate(tA, tB, bins, normalize=True, norm_bin_width=False)
    nrmn = fcs.normalization_factor(tA, tB, bins, normalize=False, norm_bin_width=True)
    # assert np.allclose(nrmn*nrmc, allc), "inconsistent correlation normalization with normalize=True, norm_bin_widht=False"
    nrmG = fcs.normalize(nrmn, tA, tB, bins, normalize=False, norm_bin_width=True)
    # assert np.allclose(nrmG, allc), "inconsistent normalize with normalize=True, norm_bin_width=False"
    
    nbwc = fcs.correlate(tA, tB, bins, normalize=False, norm_bin_width=True)
    nbwn = fcs.normalization_factor(tA, tB, bins, normalize=True, norm_bin_width=False)
    # assert np.allclose(nbwn*nbwc, allc), "inconsistent correlation normalization with normalize=False, norm_bin_widht=True"
    nbwG = fcs.normalization_factor(nbwn, tA, tB, bins, normalize=True, norm_bin_width=False)
    # assert np.allclose(nbwG, allc), "inconsistent normalize with normalize=False, norm_bin_width=True"
    
    # assert ref_tA == grc(tA), "refcount for A changed"
    # assert ref_tB == grc(tB), "refcount for B changed"
    
def test_weights(weights_types):
    tA, tB, wA, wB = weights_types
    ref_tA, ref_tB, ref_wA, ref_wB = grc(tA), grc(tB), grc(wA), grc(wB)
    wcorrl = fcs.correlate(tA, tB, bins, wA, wB)
    assert wcorrl.shape[-1] == (bins.size-1)
    # assert ref_tA == grc(tA), "refcount of timesT changed"
    # assert ref_tB == grc(tB), "refcount of timesU changed"
    # assert ref_wA == grc(wA), "refcount of weigthsT changed"
    # assert ref_wB == grc(wB), "refount of weightsU changed"
    
def test_weights_nanos(nanos_types):
    tA, tB, nA, nB = nanos_types
    ref_tA, ref_tB, ref_nA, ref_nB = grc(tA), grc(tB), grc(nA), grc(nB)
    wcorrl = fcs.correlate(tA, tB, bins, weightsconst, weightsconst, nA, nB)
    assert wcorrl.shape[-1] == (bins.size -1), "Wrong last dim of correlation"
    # assert ref_tA == grc(tA), "refcount of timesT changed"
    # assert ref_tB == grc(tA), "refcount of timesU changed"
    # assert ref_nA == grc(nA), "refcount of nanosT changed"
    # assert ref_nB == grc(nB), "refcount of nanosU changed"
