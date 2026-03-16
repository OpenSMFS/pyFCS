#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:42:38 2024

@author: paul
"""

import numpy as np
import pytest
import numba
from itertools import chain, product, permutations, combinations
import pyFCS as fcs

# accelerate generate random number according to distribtion
@numba.jit((numba.float64, numba.float64[:]))
def frac_to_dist(frac, dist):
    i = 0
    while i < dist.size and frac > dist[i]:
        i += 1
    return i

def cpd(data):
    if isinstance(data, (list, tuple)):
        return type(data)(cpd(d) for d in data)
    elif isinstance(data, np.ndarray):
        if data.dtype == object:
            out = np.empty(data.shape, dtype=object)
            for idx in product(*(range(i) for i in data.shape)):
                out[idx] = cpd(data[idx])
            return out
        else:
            return data.copy()
    else:
        return data

rng = np.random.default_rng(seed=1549)

def gen_time(center:float, sigma:float, lam:int)->np.ndarray:
    """
    generate a single array of times
    """
    return np.sort(np.random.normal(center, sigma, size=np.random.poisson(lam))).astype(np.int64)

def gen_dist(size:int, amps:np.ndarray, taus:np.ndarray, mu:float, sigma:float)->np.ndarray:
    x = np.arange(0,size,1)
    if amps.shape != amps.shape and amps.ndim == 1:
        raise ValueError("amps and taus must be same shape")
    decay = np.array([a*np.exp(-x/t)/t for a, t in zip(amps, taus)]).sum(axis=0)
    irf = np.exp(-(x-mu)**2/(2*sigma**2))
    cdf = np.convolve(decay, irf)[:size]
    return cdf / cdf.sum()

def gen_nanos(times:np.ndarray, cdf:np.ndarray)->np.ndarray:
    return np.array([frac_to_dist(rng.random(),cdf) for _ in times], dtype=np.uint16)

def gen_weights(nanos, *args):
    M = np.vstack([arg/arg.sum() for arg in args])
    nanodist = np.bincount(nanos, minlength=M.shape[1])
    nanodist[nanodist==0] = 1
    dIinv = 1/nanodist
    Mi = M*dIinv
    U = np.linalg.inv(Mi@M.T) @ Mi
    return U


def gen_singleseq(nburst:int, tsigma:tuple[float], tlam:tuple[float], tcspc_nb:int, taus:tuple[np.ndarray], mus:tuple[float], sigmas:tuple[float]):
    for nb in nburst:
        ta = gen_time()

def times_test(timesA, timesB, bins, kwargs):
    timesT, timesU, bns = cpd(timesA), cpd(timesB), cpd(bins)
    out = fcs.correlate(timesT, timesU, bns, **kwargs)
    out = fcs.correlate(timesT, timesU, bns, **kwargs)
    out += 12
    timesT += 1
    timesU += 2
    bns += 1
    bns -= 1
    out = fcs.correlate(timesT, timesU, bns)
    out = fcs.correlate(timesT, timesU, bns, **kwargs)

def weights_test(twA, twB, bins, kwargs):
    timesT, weightsT = twA
    timesU, weightsU = twB
    timesT, timesU, weightsT, weightsU, bns = cpd(timesT), cpd(timesU), cpd(weightsT), cpd(weightsU), cpd(bins)
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU, **kwargs)
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU, **kwargs)
    out += 12
    timesT += 1
    timesU += 2
    weightsT += 1
    weightsU += 1
    weightsT -= 1
    weightsU -= 1
    bns += 1
    bns -= 1
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU)
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU, **kwargs)

def weightsnanos_test(twnA, twnB, bins, kwargs):
    timesT, weightsT, nanosT = twnA
    timesU, weightsU, nanosU = twnB
    timesT, timesU, weightsT, weightsU, bns = cpd(timesT), cpd(timesU), cpd(weightsT), cpd(weightsU), cpd(bins)
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU, **kwargs)
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU, **kwargs)
    out += 12
    timesT += 1
    timesU += 2
    weightsT += 1
    weightsU += 1
    weightsT -= 1
    weightsU -= 1
    bns += 1
    bns -= 1
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU)
    out = fcs.correlate(timesT, timesU, bns, weightsT=weightsT, weightsU=weightsU, **kwargs)

