#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:59:42 2026

@author: paul
"""
from numbers import Number
from itertools import permutations, product
import numpy as np

import pytest

import pyFCS as fcs


@pytest.fixture
def expect_data():
    ta = np.array([ 10,  11,  42,  62,  87, 137, 195, 219, 309, 317, 331, 346, 372], dtype=np.uint64)
    tb = np.array([ 15,  49,  73,  83, 108, 158, 216, 240, 330, 338, 352, 393], dtype=np.uint64)
    bins = np.array([1, 10, 20, 40, 80], dtype=np.uint64)
    
    out = {
        'a':{'arr':ta, 
             'a':np.array([  2,  2,  9, 11], dtype=np.uint64), 
             'b':np.array([  5,  2, 14, 14], dtype=np.uint64)},
        'b':{'arr':tb,
             'a':np.array([  4,  3,  6,  8], dtype=np.uint64), 
             'b':np.array([  1,  2,  7,  9], dtype=np.uint64)},
        'bins':bins
        }
    return out
    

@pytest.fixture(params=list(product(('a', 'b'), ('a', 'b'))))
def expect_times(expect_data, request):
    if not isinstance(request.param, bool):
        T, U = request.param
        tmT, tmU = expect_data[T]['arr'], expect_data[U]['arr']
        expect = expect_data[T][U]
    elif request.param:
        tmT = [expect_data['a']['arr'], expect_data['b']['arr']]
        tmU = [expect_data['b']['arr'], expect_data['a']['arr']]
        expect = expect_data['a']['b'] + expect_data['b']['a']
    else:
        tmT = [expect_data['a']['arr'], expect_data['b']['arr']]
        tmU = [expect_data['a']['arr'], expect_data['b']['arr']]
        expect = expect_data['a']['a'] + expect_data['b']['b']
    return tmT, tmU, expect_data['bins'], expect

def test_correlate_expect(expect_times):
    timesT, timesU, bins, expect = expect_times
    corrl = fcs.correlate(timesT, timesU, bins, normalize=False, norm_bin_width=False)
    assert np.all(corrl==expect), f"Correlation histogram incorect for basic algorithm, {corrl}, {expect}"




def test_correlate_nano_dummy_expect(expect_times):
    def make_nanos(times):
        if isinstance(times[0], Number):
            return np.arange(len(times), dtype=np.uint8) % 2
        return [make_nanos(t) for t in times]
    
    timesT, timesU, bins, expect = expect_times
    weights = np.ones(2, dtype=np.float64)
    nanosT = make_nanos(timesT)
    nanosU = make_nanos(timesU)
    corrl = fcs.correlate(timesT, timesU, bins, 
                          weightsT=weights, weightsU=weights,
                          nanosT=nanosT, nanosU=nanosU,
                          normalize=False, norm_bin_width=False)
    assert np.all(corrl==expect), f"Correlation histogram incorect for all ones weighted nanos, {corrl}, {expect}"
    corrl = fcs.correlate(timesT, timesU, bins, 
                          weightsT=weights*2, weightsU=weights*2,
                          nanosT=nanosT, nanosU=nanosU,
                          normalize=False, norm_bin_width=False)
    assert np.allclose(corrl, expect*4.0), f"Correlation histogram incorect for all ones weighted nanos, {corrl}, {expect*4.0}"


def test_correlate_weight_dummy_expect(expect_times):
    def make_weights(times):
        if isinstance(times[0], Number):
            return np.ones(len(times), dtype=np.float64)
        return [make_weights(t) for t in times]
    timesT, timesU, bins, expect = expect_times
    weightsT = make_weights(timesT)
    weightsU = make_weights(timesU)
    corrl = fcs.correlate(timesT, timesU, bins, 
                          weightsT=weightsT, weightsU=weightsU,
                          normalize=False, norm_bin_width=False)
    assert np.all(corrl==expect), f"Correlation histogram incorect for all ones weighted nanos, {corrl}, {expect}"
    corrl = fcs.correlate(timesT, timesU, bins, 
                          weightsT=weightsT*2, weightsU=weightsU*2,
                          normalize=False, norm_bin_width=False)
    assert np.allclose(corrl, expect*4.0), f"Correlation histogram incorect for all ones weighted nanos, {corrl}, {expect*4.0}"


@pytest.fixture(params=[False, True])
def expect_nanos(expect_data, request):
    def expect_sub(expect_data, a, b):
        tT, tU = expect_data[a]['arr'], expect_data[b]['arr']
        times = np.concatenate([tT, tU])
        nanos = np.zeros(times.size+tU.size, dtype=np.uint8)
        nanos[tT.size:] = 1
        sort = np.argsort(times)
        times = times[sort]
        nanos = nanos[sort]
        sum_arr = [(a,), (b,), (a, b)]
        expect = np.array([[sum(expect_data[aa][bb] for aa, bb in 
                                product(sum_arr[i], sum_arr[j])) 
                            for j in range(3)] for i in range(3)]).astype(np.float64)
        return times, nanos, expect
    weights = np.array([[1.0, 0.0],[0.0, 1.0], [1.0, 1.0]])
    if request.param:
        times, nanos, e = zip(expect_sub(expect_data, 'a', 'b'), 
                                       expect_sub(expect_data, 'b', 'a'))
        return times, expect_data['bins'], weights, nanos, sum(e)
    times, nanos, expect = expect_sub(expect_data, 'a', 'b')
    return times, expect_data['bins'], weights, nanos, expect


@pytest.fixture
def expect_weights(expect_nanos):
    times, bins, weights, nanos, expect = expect_nanos
    weights = weights[:,nanos] if isinstance(nanos[0], Number) else [weights[:,n] for n in nanos]
    return times, bins, weights, expect


def test_correlate_nanos_expect(expect_nanos):
    times, bins, weights, nanos, expect = expect_nanos
    corr_cross = fcs.correlate(
        times, times, bins, weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos,
        normalize=False, norm_bin_width=False, cross_correlate=True
                             )
    assert np.all(corr_cross == expect), "nanos weighted correlation incorrect"
    corr_all = fcs.correlate(
        times, times, bins, weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos,
        normalize=False, norm_bin_width=False, cross_correlate=False
                             )
    assert np.all(corr_cross[range(3), range(3)] == corr_all), "cross and simple correlations do not match"
    for a, b in permutations(range(3), 2):
        corr_temp = fcs.correlate(times, times, bins, 
                                  weightsT=weights[a], weightsU=weights[b],
                                  nanosT=nanos, nanosU=nanos, 
                                  normalize=False, norm_bin_width=False)
        # evaluate expected value at given pos
        assert np.all(corr_cross[a,b] == corr_temp), "weights nanos subcorrelation different from cross"


def test_correlate_weights_expect(expect_weights):
    times, bins, weights, expect = expect_weights
    corr_cross = fcs.correlate(
        times, times, bins, weightsT=weights, weightsU=weights,
        normalize=False, norm_bin_width=False, cross_correlate=True
                             )
    assert np.all(corr_cross == expect), "nanos weighted correlation incorrect"
    corr_all = fcs.correlate(
        times, times, bins, weightsT=weights, weightsU=weights,
        normalize=False, norm_bin_width=False, cross_correlate=False
                             )
    assert np.all(corr_cross[range(3), range(3)] == corr_all), "cross and simple correlations do not match"
    for a, b in permutations(range(3), 2):
        wa = weights[a] if isinstance(times[0], Number) else [w[a] for w in weights]
        wb = weights[b] if isinstance(times[0], Number) else [w[b] for w in weights]
        corr_temp = fcs.correlate(times, times, bins, weightsT=wa, weightsU=wb,
                                  normalize=False, norm_bin_width=False)
        # evaluate expected value at given pos
        assert np.all(corr_cross[a,b] == corr_temp), "weights nanos subcorrelation different from cross"


@pytest.fixture(params=[(nrm, nbw, mz) for nrm, nbw, mz in 
                        product(*((False, True) for _ in range(3)))
                        if nrm or nbw])
def kwarg_norm(request):
    return request.param


def get_edge(tT, tU, mz):
    if isinstance(tT[0], Number):
        mn = 0 if mz else min((tT[0], tU[0]))
        mx = max((tT[-1], tU[-1]))
        return np.array([mn, mx], dtype=np.uint64)
    return [get_edge(t, u, mz) for t, u in zip(tT, tU)]


def test_edges(expect_times, kwarg_norm):
    timesT, timesU, bins, _ = expect_times
    nrm, nbw, mz = kwarg_norm
    corrl_a = fcs.correlate(timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, minzero=mz)
    edge = get_edge(timesT, timesU, mz)
    try:
        corrl_b = fcs.correlate(timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, edges=edge)
    except:
        raise ValueError(str(edge))
    assert np.allclose(corrl_a, corrl_b), f"correlate edges vs minzero do not match, {corrl_a}, {corrl_b}"


def test_normalization_factor(expect_times, kwarg_norm):
    timesT, timesU, bins, _ = expect_times
    nrm, nbw, mz = kwarg_norm
    edge = get_edge(timesT, timesU, mz)
    corr_raw = fcs.correlate(timesT, timesU, bins, normalize=False, norm_bin_width=False)
    corrl_a = fcs.correlate(timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, minzero=mz)
    nrf_a =  fcs.normalization_factor(timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, minzero=mz)
    assert np.allclose(corr_raw*nrf_a, corrl_a), "raw vs normalization constant do not match"
    nrf_b =  fcs.normalization_factor(timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, edges=edge)
    assert np.all(nrf_a == nrf_b), "normalization constant edges vs minzero do not match"


def test_normalize(expect_times, kwarg_norm):
    timesT, timesU, bins, _ = expect_times
    nrm, nbw, mz = kwarg_norm
    edge = get_edge(timesT, timesU, mz)
    corr_raw = fcs.correlate(timesT, timesU, bins, normalize=False, norm_bin_width=False)
    corrl_a = fcs.correlate(timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, minzero=mz)
    corrln_a = fcs.normalize(corr_raw, timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, minzero=mz)
    assert np.allclose(corrln_a, corrl_a), "normalize vs correlate mismatch"
    corrln_b = fcs.normalize(corr_raw, timesT, timesU, bins, normalize=nrm, norm_bin_width=nbw, edges=edge)
    assert np.all(corrln_a == corrln_b), "normalize vs correlate edges vs minzero do not match"


def test_normalize_nanos(expect_nanos, kwarg_norm):
    times, bins, weights, nanos, expect = expect_nanos
    nrm, nbw, mz = kwarg_norm
    edge = get_edge(times, times, mz)
    corr_raw = fcs.correlate(times, times, bins, 
                             weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos,
                             normalize=False, norm_bin_width=False, cross_correlate=True)
    corrl_a = fcs.correlate(times, times, bins, 
                            weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos,
                            normalize=nrm, norm_bin_width=nbw, minzero=mz, cross_correlate=True)
    corrln_a = fcs.normalize(corr_raw, times, times, bins, 
                             weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos,
                             normalize=nrm, norm_bin_width=nbw, minzero=mz)
    assert np.allclose(corrln_a, corrl_a), "normalize vs correlate mismatch"
    corrln_b = fcs.normalize(corr_raw, times, times, bins, 
                             weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos,
                             normalize=nrm, norm_bin_width=nbw, edges=edge)
    assert np.all(corrln_a == corrln_b), "normalize vs correlate edges vs minzero do not match"


def test_normalization_weights(expect_nanos, kwarg_norm):
    times, bins, weights, nanos, expect = expect_nanos
    nrm, nbw, mz = kwarg_norm
    edge = get_edge(times, times, mz)
    corr_raw = fcs.correlate(times, times, bins, 
                             weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos, 
                             normalize=False, norm_bin_width=False, cross_correlate=True)
    corrl_a = fcs.correlate(times, times, bins, 
                            weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos, 
                            normalize=nrm, norm_bin_width=nbw, minzero=mz, cross_correlate=True)
    nrf_a =  fcs.normalization_factor(times, times, bins, 
                                      weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos, 
                                      normalize=nrm, norm_bin_width=nbw, minzero=mz, cross_correlate=True)
    assert np.allclose(corr_raw*nrf_a, corrl_a), f"raw vs normalization constant do not match, {corr_raw*nrf_a - corrl_a}"
    nrf_b =  fcs.normalization_factor(times, times, bins, 
                                      weightsT=weights, weightsU=weights, nanosT=nanos, nanosU=nanos, 
                                      normalize=nrm, norm_bin_width=nbw, edges=edge, cross_correlate=True)
    assert np.all(nrf_a == nrf_b), "normalization constant edges vs minzero do not match"