#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:57:50 2026

@author: paul
"""
from sys import getrefcount as grc
from itertools import chain, product
from numbers import Number

import numpy as np

import pyFCS as fcs

import pytest


@pytest.fixture(params=[(nrm, nbw, mz) for nrm, nbw, mz in 
                        product(*((False, True) for _ in range(3)))])
def kwarg_norm(request):
    return request.param


def get_edge(tT, tU, mz, tp):
    if isinstance(tT[0], Number):
        mn = 0 if mz else min((tT[0], tU[0]))
        mx = max((tT[-1], tU[-1]))
        return np.array([mn, mx], dtype=tp)
    return [get_edge(t, u, mz, tp) for t, u in zip(tT, tU)]


def check_sub(val):
    if isinstance(val, np.ndarray):
        return val.dtype == np.object_
    return not isinstance(val[0], Number)


def refcounts(val):
    if check_sub(val):
        return val, tuple(refcounts(v) for v in val)
    return grc(val)


def refcount_map(args, kwargs):
    kwarg_ord = sorted(kwargs.keys())
    return tuple(refcounts(arg) for arg in args) + tuple(refcounts(kwargs[k]) for k in kwarg_ord)


def check_refcount(func, args, kwargs, skip):
    rmap = refcount_map(args, kwargs)
    out = func(*args, **{k:v for k, v in chain(kwargs.items(), skip.items())})
    assert rmap == refcount_map(args, kwargs), 'refcounts changed'
    return out


def test_correlate(kwarg_norm):
    types_ = (np.int64, np.uint64)
    nrm, nbw, mz = kwarg_norm
    kw = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz}
    kwe = {'normalize':nrm, 'norm_bin_width':nbw}
    for t_tT, t_tU, t_bin, t_edge in product(types_, types_, types_, types_):
        bins = np.array([1, 10, 20, 30, 40], dtype=t_bin)
        tT = np.cumsum(np.random.poisson(40, 50)).astype(t_tT)
        tU = np.cumsum(np.random.poisson(40, 50)).astype(t_tU)
        
        check_refcount(fcs.correlate, (tT, tT, bins), {}, kw)
        check_refcount(fcs.correlate, (tT, tU, bins), {}, kw)
        
        if nrm or nbw:
            check_refcount(fcs.normalization_factor, (tT, tT, bins), {}, kw)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tT, bins), {}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), {}, kw)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tU, bins), {}, kw)
            
            edgeTT = get_edge(tT, tT, mz, t_edge)
            edgeTU = get_edge(tT, tU, mz, t_edge)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), {'edges':edgeTT}, kwe)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tT, bins), {'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), {'edges':edgeTU}, kwe)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tU, bins), {'edges':edgeTU}, kwe)


def test_correlate_weights(kwarg_norm):
    types_ = (np.int64, np.uint64)
    nrm, nbw, mz = kwarg_norm
    kw = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz}
    kwc = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz, 'cross_correlate':True}
    kwe = {'normalize':nrm, 'norm_bin_width':nbw}
    kwec = {'normalize':nrm, 'norm_bin_width':nbw, 'cross_correlate':True}
    for t_tT, t_tU, t_wT, t_wU, t_bin, t_edge in product(*(types_ for _ in range(6))):
        bins = np.array([1, 10, 20, 30, 40], dtype=t_bin)
        tT = np.cumsum(np.random.poisson(40, 50)).astype(t_tT)
        tU = np.cumsum(np.random.poisson(30, 60)).astype(t_tU)
        wT = np.random.random((3, tT.size))
        wU = np.random.random((3, tU.size))

        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT}, 
                       kw)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU}, 
                       kw)
        
        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT}, 
                       kwc)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU}, 
                       kwc)
        
        if nrm or nbw:
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kwc)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kwc)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kw)
            
            
            edgeTT = get_edge(tT, tT, mz, t_edge)
            edgeTU = get_edge(tT, tU, mz, t_edge)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwec)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwec)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwe)


def test_correlate_nanos(kwarg_norm):
    types_ = (np.int64, np.uint64)
    nrm, nbw, mz = kwarg_norm
    kw = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz}
    kwc = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz, 'cross_correlate':True}
    kwe = {'normalize':nrm, 'norm_bin_width':nbw}
    kwec = {'normalize':nrm, 'norm_bin_width':nbw, 'cross_correlate':True}
    for t_tT, t_tU, t_wT, t_wU, t_bin, t_edge in product(*(types_ for _ in range(6))):
        bins = np.array([1, 10, 20, 30, 40], dtype=t_bin)
        tT = np.cumsum(np.random.poisson(40, 50)).astype(t_tT)
        tU = np.cumsum(np.random.poisson(30, 60)).astype(t_tU)
        wT = np.random.random((3, 5))
        wU = np.random.random((3, 5))
        nT = np.random.randint(0,5, tT.size)
        nU = np.random.randint(0,5, tU.size)

        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, 
                       kw)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, 
                       kw)
        
        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, 
                       kwc)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, 
                       kwc)
        
        if nrm or nbw:
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kwc)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kwc)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kw)
            
            
            edgeTT = get_edge(tT, tT, mz, t_edge)
            edgeTU = get_edge(tT, tU, mz, t_edge)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU,
                            'edges':edgeTU}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU, 
                            'edges':edgeTU}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwec)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU, 
                            'edges':edgeTU}, kwec)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU, 
                            'edges':edgeTU}, kwe)


def test_correlate_multi(kwarg_norm):
    types_ = (np.int64, np.uint64)
    nrm, nbw, mz = kwarg_norm
    kw = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz}
    kwe = {'normalize':nrm, 'norm_bin_width':nbw}
    for t_tT, t_tU, t_bin, t_edge in product(types_, types_, types_, types_):
        bins = np.array([1, 10, 20, 30, 40], dtype=t_bin)
        tT = [np.cumsum(np.random.poisson(40, s)).astype(t_tT) for s in (70, 60, 50)]
        tU = [np.cumsum(np.random.poisson(30, s)).astype(t_tU) for s in (60, 70, 50)]
        
        check_refcount(fcs.correlate, (tT, tT, bins), {}, kw)
        check_refcount(fcs.correlate, (tT, tU, bins), {}, kw)
        
        if nrm or nbw:
            check_refcount(fcs.normalization_factor, (tT, tT, bins), {}, kw)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tT, bins), {}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), {}, kw)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tU, bins), {}, kw)
            
            edgeTT = get_edge(tT, tT, mz, t_edge)
            edgeTU = get_edge(tT, tU, mz, t_edge)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), {'edges':edgeTT}, kwe)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tT, bins), {'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), {'edges':edgeTU}, kwe)
            check_refcount(fcs.normalize, (np.ones(bins.size-1), tT, tU, bins), {'edges':edgeTU}, kwe)


def test_correlate_weights_multi(kwarg_norm):
    types_ = (np.int64, np.uint64)
    nrm, nbw, mz = kwarg_norm
    kw = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz}
    kwc = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz, 'cross_correlate':True}
    kwe = {'normalize':nrm, 'norm_bin_width':nbw}
    kwec = {'normalize':nrm, 'norm_bin_width':nbw, 'cross_correlate':True}
    for t_tT, t_tU, t_wT, t_wU, t_bin, t_edge in product(*(types_ for _ in range(6))):
        bins = np.array([1, 10, 20, 30, 40], dtype=t_bin)
        tT = [np.cumsum(np.random.poisson(40, s)).astype(t_tT) for s in (70, 60, 50)]
        tU = [np.cumsum(np.random.poisson(30, s)).astype(t_tU) for s in (60, 70, 50)]
        wT = [np.random.random((3, t.size)) for t in tT]
        wU = [np.random.random((3, t.size)) for t in tU]

        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT}, 
                       kw)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU}, 
                       kw)
        
        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT}, 
                       kwc)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU}, 
                       kwc)
        
        if nrm or nbw:
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kwc)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kwc)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU}, kw)
            
            
            edgeTT = get_edge(tT, tT, mz, t_edge)
            edgeTU = get_edge(tT, tU, mz, t_edge)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwec)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwec)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'edges':edgeTU}, kwe)


def test_correlate_nanos_multi(kwarg_norm):
    types_ = (np.int64, np.uint64)
    nrm, nbw, mz = kwarg_norm
    kw = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz}
    kwc = {'normalize':nrm, 'norm_bin_width':nbw, 'minzero':mz, 'cross_correlate':True}
    kwe = {'normalize':nrm, 'norm_bin_width':nbw}
    kwec = {'normalize':nrm, 'norm_bin_width':nbw, 'cross_correlate':True}
    for t_tT, t_tU, t_wT, t_wU, t_bin, t_edge in product(*(types_ for _ in range(6))):
        bins = np.array([1, 10, 20, 30, 40], dtype=t_bin)
        tT = [np.cumsum(np.random.poisson(40, s)).astype(t_tT) for s in (70, 60, 50)]
        tU = [np.cumsum(np.random.poisson(30, s)).astype(t_tU) for s in (60, 70, 50)]
        wT = np.random.random((3, 5))
        wU = np.random.random((3, 5))
        nT = [np.random.randint(0,5, t.size) for t in tT]
        nU = [np.random.randint(0,5, t.size) for t in tU]

        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, 
                       kw)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, 
                       kw)
        
        check_refcount(fcs.correlate, (tT, tT, bins), 
                       {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, 
                       kwc)
        check_refcount(fcs.correlate, (tT, tU, bins), 
                       {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, 
                       kwc)
        
        if nrm or nbw:
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kw)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kwc)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT}, kw)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kwc)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU}, kw)
            
            
            edgeTT = get_edge(tT, tT, mz, t_edge)
            edgeTU = get_edge(tT, tU, mz, t_edge)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU,
                            'edges':edgeTU}, kwe)
            check_refcount(fcs.normalize, (np.ones((3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU, 
                            'edges':edgeTU}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwec)
            check_refcount(fcs.normalize, (np.ones((3, 3, bins.size-1)), tT, tT, bins), 
                           {'weightsT':wT, 'weightsU':wT, 'nanosT':nT, 'nanosU':nT, 
                            'edges':edgeTT}, kwe)
            
            check_refcount(fcs.normalization_factor, (tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU, 
                            'edges':edgeTU}, kwec)
            check_refcount(fcs.normalize, (np.ones((3,3, bins.size-1)), tT, tU, bins), 
                           {'weightsT':wT, 'weightsU':wU, 'nanosT':nT, 'nanosU':nU, 
                            'edges':edgeTU}, kwe)
            