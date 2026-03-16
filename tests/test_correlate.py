#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Paul David Harris
# Purpose: test modules for pyFCS
"""
Testing moduldes for pyFCS package.
"""
from sys import getrefcount as grc
from functools import wraps
import numpy as np
import numba
import pytest
from itertools import product

import pyFCS as fcs

rng = np.random.default_rng(158997)

def check_refcount(func):
    @wraps(func)
    def inner(*args, **kwargs):
        arg_counts = tuple(grc(a) for a in args)
        kwarg_counts = {key:grc(val) for key, val in kwargs.items()}
        out = func(*args, **kwargs)
        
        oarg_counts = tuple(grc(a) for a in args)
        okwarg_counts = {key:grc(val) for key, val in kwargs.items()}
        d_args = tuple(b-a for a, b in zip(arg_counts, oarg_counts))
        print(arg_counts, oarg_counts)
        print(kwarg_counts, okwarg_counts)
        k_kwargs = dict()
        for i, d in enumerate(arg_counts):
            if d != oarg_counts[i]:
                print(f'arg[{i}] changed, started refcoutn {d} but ended {oarg_counts[i]}')
        for key in kwarg_counts.keys():
            if key not in okwarg_counts:
                print(f"kwargs {key} dropped from output of kwargs, kwargs modified")
                continue
            k_kwargs[key] = okwarg_counts[key] - kwarg_counts[key]
            if kwarg_counts[key] != okwarg_counts[key]:
                
                print(f'kwargs[{key}] changed, started {kwarg_counts[key]}, but ended {okwarg_counts[key]}')
        for key in okwarg_counts.keys():
            if key not in kwarg_counts:
                print(f"{key} added to kwargs by function, kwargs modified")
        out_rcnt = grc(out)
        print(out_rcnt)
        return out, d_args, k_kwargs, out_rcnt
    return inner

@numba.jit((numba.float64[:], numba.float64[:]))
def frac_to_dist(frac, dist):
    out = np.empty(frac.size, dtype=np.int16)
    for n, f in enumerate(frac):
        i = 0
        while i < dist.size and f > dist[i]:
            i += 1
        out[n] = i
    return out

def make_times_data(n:int):
    outA = np.empty(n, dtype=object)
    outB = np.empty(n, dtype=object)
    for i in range(n):
        loc  = rng.integers(0,int(1e9))
        outA[i] = np.sort(rng.normal(loc, 1000, rng.poisson(100))).astype(np.int64)
        outB[i] = np.sort(rng.normal(loc, 1000, rng.poisson(100))).astype(np.int64)
    return outA, outB

def make_decay(sz:int, lt:float, pos:float, sigma:float):
    x = np.arange(sz)
    y = np.cumsum(np.convolve(np.exp(-lt*x), np.exp(-((x-pos)**2)/(2*np.pi*sigma)), mode='full')[:sz])
    y /= y.sum()
    return y

def make_nanos(times, decay):
    nano = np.empty(times.size, dtype=object)
    for i in range(times.size):
        nano[i] = frac_to_dist(rng.random(times[i].size), decay)
    return nano

def make_weights_list(weight, nanos):
    weights = np.empty(nanos.size, dtype=object)
    for i in range(nanos.size):
        weights[i] = weight[:,nanos[i]]
    return weights
    
def extract_normalizations(**kwargs):
    normalize = not kwargs.get('normalize', True)
    norm_bin_width = not kwargs.get('norm_bin_width', True)
    min_zero = kwargs.get('minzero', False)
    return normalize, norm_bin_width, min_zero

def cast_int16(inp):
    return inp.astype(np.int16)

def cast_uint16(inp):
    return inp.astype(np.uint16)

def cast_int64(inp):
    return inp.astype(np.int64)

def cast_uint64(inp):
    return inp.astype(np.uint64)

def cast_tuple(inp):
    if np.issubdtype(inp.dtype, np.integer):    
        return tuple(int(i) for i in inp)
    else:
        return tuple(float(i) for i in inp)

def cast_list(inp):
    if np.issubdtype(inp.dtype, np.integer):    
        return list(int(i) for i in inp)
    else:
        return list(float(i) for i in inp)

def outer_tuple(inp, inner):
    return tuple(inner(arr) for arr in inp), None

def outer_list(inp, inner):
    return list(inner(arr) for arr in inp), None

def outer_ndarray(inp, inner):
    out = np.empty(inp.shape, dtype=object)
    for i in range(inp.size):
        out[i] = inner(inp[i])
    return out, None

def outer_concatenate(inp, inner):
    out = np.concatenate(inp)
    sort = np.argsort(out)
    return inner(out[sort]), sort
    

def outer_concatenate_order(inp, inner, order):
    out = np.concatenate(inp)[order]
    return inner(out)

def gen_weights(nanos, *args):
    if not isinstance(nanos, np.ndarray) and np.issubdtype(nanos.dtype, np.number):
        if np.issubdtype(type(nanos[0]), np.number):
            nanos = np.array(nanos, dtype=np.int16)
        else:
            nanos = np.concatenate(nanos, dtype=np.int16)
    M = np.vstack([arg/arg.sum() for arg in args])
    nanodist = np.bincount(nanos, minlength=M.shape[1])
    nanodist[nanodist==0] = 1
    dIinv = 1/nanodist
    Mi = M*dIinv
    U = np.linalg.inv(Mi@M.T) @ Mi
    return U

def compute_edge(A, B, offset=2):
    """Compute, for a single pair of input arrays, the edges, with offset to
    add to edge to shift the start/stop"""
    if isinstance(offset, str) and offset == 'rand':
        offset = np.random.randint(0, 100, size=2)
        offset[0] = -offset[0]
    elif np.issubdtype(type(offset), np.integer):
        offset = np.array([-offset, offset], dtype=np.int64)
    edg = np.array([min(np.min(A), np.min(B)), max(np.max(A), np.max(B))], dtype=np.int64)
    edg += offset
    edg[edg<0] = 0
    return edg

def compute_edges(A, B, offset=2):
    if np.issubdtype(type(A[0]), np.number):
        return compute_edge(A,B, offset=offset)
    return np.array([compute_edge(a, b, offset=offset) for a, b in zip(A, B)])

# data styles:
# 1. bursts concatenated
# 2. bursts separate
@pytest.fixture(scope='module', params=[np.int64, np.uint64])
def bins(request):
    return np.logspace(0,5,10, dtype=request.param)

@pytest.fixture(scope='module', params=tuple(product(["None", False, True],
                                                    ["None", False, True],
                                                    ["None", False, True], 
                                                    ["None", False, True],
                                                    ["None", 1, 8])))
def kwargs(request)->dict:
    return {key:False for key, val in zip(['validate', 'normalize', 'norm_bin_width', 'minzero', 'max_cores'], request.param) if  val != "None"}


timesAB = make_times_data(20)
edgesAB = (compute_edge(timesAB[0], timesAB[0]), compute_edge(timesAB[0], timesAB[1]))
decays = (make_decay(256, 60, 20, 5), make_decay(256, 60, 20, 5), np.ones(256)/256)
nanosAB = (make_nanos(timesAB[0], decays[0]), make_nanos(timesAB[1], decays[1]))

time_combs = tuple(product([outer_ndarray, outer_list, outer_tuple, outer_concatenate],
                           [cast_int64, cast_uint64, cast_list, cast_tuple]))
nano_combs = tuple(product([outer_ndarray, outer_list, outer_tuple, outer_concatenate_order], 
                           [cast_int16, cast_uint16, cast_list, cast_tuple, cast_int64]))

time_combsAB = tuple( A + B + (U, ) for A, B, U in product(time_combs, time_combs, range(2)) 
                     if (A[0] is outer_concatenate) == (B[0] is outer_concatenate))[:20]
nano_combsAB = tuple(A+B+ (U, ) for A, B, U in product(nano_combs, nano_combs, range(2))
                     if (A[0] is outer_concatenate_order) == (B[0] is outer_concatenate_order))[:20]


@pytest.fixture(scope='module', params=time_combsAB)
def times_fixture(request):
    outerA, innerA, outerB, innerB, U = request.param
    timesA, _ = outerA(timesAB[0], innerA)
    if U == 0 and outerA is outerB and innerA is innerB:
        return timesA, timesA
    return timesA, outerB(timesAB[U], innerB)[0]

@pytest.fixture(scope='module', params=time_combsAB)
def times_weights_nanos(request):
    outerA, innerA, outerB, innerB, U = request.param
    timesA, timesB  = timesAB[0], timesAB[U], 
    nanosA, nanosB = nanosAB[0], nanosAB[U]
    if (outerA is outer_concatenate_order):
        timesA, orderA = outer_concatenate(timesA, cast_int64)
    else:
        orderA = None
    nanosA = outerA(nanosA, innerA, orderA)
    if U == 0 and outerA is outerB and innerA is innerB:
        timesB, nanosB = timesA, nanosA
    else:
        if outerB is outer_concatenate_order:
            timesB, orderB = outer_concatenate(timesB, cast_int64)
        else:
            orderB = None
        nanosB = outerB(nanosB, innerB, orderB)
    weightsA = gen_weights(np.concatenate([nanosA, nanosB]), *decays)
    weightsB = weightsA
    return timesA, timesB, weightsA, weightsB, nanosA, nanosB


def merge_dict(*args):
    out = dict()
    for arg in args:
        for key, value in arg.items():
            out[key] = value
    return out

@pytest.fixture(scope='module', params=(product([(outer_ndarray, outer_ndarray), (outer_concatenate, outer_concatenate_order)], [0,1,2], [False, True])))
def data_single_double(request):
    outer_castT, out_sort, edges = request.param
    timesT, timesU = timesAB
    out = {'timesT':timesT, 'timesU':timesU}
    if out_sort > 0:
        nanosT, nanosU = nanosAB
        weights = gen_weights(np.concatenate([nanosT, nanosU]), *decays)
    if out_sort == 1:
        out.update(weightsT=weights, weightsU=weights, nanosT=nanosT, nanosU=nanosU)
    elif out_sort == 2:
        out['weightsT'] = make_weights_list(weights, nanosT)
        out['weightsu'] = make_weights_list(weights, nanosU)
    if edges:
        out['edges'] = compute_edges(timesT, timesU)
    return out
    

def test_kwargs(data_single_double, kwargs):
    pass
    
def test_correlate_times(times_fixture):
    timesT, timesU = times_fixture
    corrl = fcs.correlate(timesT, timesU, bins, **kwargs)
    assert corrl.size == bins.size -1, "Wrong size of correlation output"
    edges = compute_edge(timesT, timesU)
    if kwargs.get('minzero', False):
        with pytest.warns(UserWarning):
            ecorrl = fcs.correlate(timesT, timesU, bins, edges=edges, **kwargs)
    else:
        ecorrl = fcs.correlate(timesT, timesU, bins, edges=edges, **kwargs)
    assert ecorrl.size == bins.size -1, "Wrong size of correlation output (edges)"
    if kwargs.get('normalize', False) or kwargs.get('norm_bin_width', False):
        ckwargs = {key:True if key in ('normalize', 'norm_bin_width') else value for key, value in kwargs.items()}
        nkwargs = {key: value if key == 'minzero' else not value for key, value in kwargs.items() if key != 'max_cores'}
        tcorrl = fcs.correlate(timesT, timesU, bins, **ckwargs)
        norm = fcs.normalization_factor(timesT, timesU, bins, **nkwargs)
        ncorrl = fcs.normalize(corrl, timesT, timesU, **nkwargs)
        assert norm.size == bins.size - 1, "wrong size for normalization factor output"
        assert np.allclose(corrl*norm, tcorrl), "Normalization factors different depending on method"
        assert np.allclose(ncorrl, tcorrl), "Normalizations different depending on method"
        if kwargs.get('minzero', False):
            with pytest.wargs(UserWarning):
                etcorrl = fcs.correlate(timesT, timesU, bins, edges=edges, **ckwargs)
            with pytest.wargs(UserWarning):
                enorm = fcs.normalization_factor(timesT, timesU, bins, edges=edges, **nkwargs)
            with pytest.wargs(UserWarning):
                encorrl = fcs.normalize(ecorrl, timesT, timesU, edges=edges, **nkwargs)
        else:
            etcorrl = fcs.correlate(timesT, timesU, bins, edges=edges, **ckwargs)
            enorm = fcs.normalization_factor(timesT, timesU, bins, edges=edges, **nkwargs)
            encorrl = fcs.normalize(ecorrl, timesT, timesU, edges=edges, **nkwargs)
        assert enorm.size == bins.size - 1, "wrong size for normalization factor output (edges)"
        assert np.allclose(ecorrl*enorm, etcorrl), "Normalization factors different depending on method (edges)"
        assert np.allclose(encorrl, etcorrl), "Normalizations different depending on method (edges)"


def test_correlate_times_nanos(times_weights_nanos, bins, kwargs):
    timesT, timesU, weightsT, weightsU, nanosT, nanosU = times_weights_nanos
    corrl = fcs.correlate(timesT, timesU, weightsT, weightsU, nanosT, nanosU, bins, **kwargs)
    assert corrl.ndim == weightsT.ndim, "Wrong dimensionality"
    assert corrl.shape[1] == bins.size -1, "Wrong size of correlation output"
    assert corrl.shape[0] == weightsT.shape[0], "Wrong number of correlations"
    for scorrl, sweightsT, sweightsU in zip(corrl, weightsT, weightsU):
        nscorrl = fcs.correlate(timesT, timesU, bins, sweightsT, sweightsU, nanosT, nanosU, **kwargs)
        assert nscorrl.ndim == 1
        assert np.allclose(scorrl, nscorrl), "multi and single weights not matching dimensions"
    if kwargs.get('normalize', False) or kwargs.get('norm_bin_width', False):
        ckwargs = {key:True if key in ('normalize', 'norm_bin_width') else value for key, value in kwargs.items()}
        nkwargs = {key: value if key == 'minzero' else not value for key, value in kwargs.items() if key != 'max_cores'}
        tcorrl = fcs.correlate(timesT, timesU, weightsT, weightsU, nanosT, nanosU, bins, **ckwargs)
        norm = fcs.normalization_factor(timesT, timesU, bins, **nkwargs)
        ncorrl = fcs.normalize(corrl, timesT, timesU, **nkwargs)
        assert norm.size == bins.size - 1, "wrong size for normalization factor output"
        assert np.allclose(corrl*norm[np.newaxis, :], tcorrl), "Normalization factors different depending on method"
        assert np.allclose(ncorrl, tcorrl), "Normalizations different depending on method"
        