#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke tests for correlate list, may later become exclusive set of tests
"""

import pytest

import numpy as np
from itertools import product, chain
import pyFCS as fcs

# setup testing arrays
rgen = np.random.RandomState(seed=433)
# generate random "times" for molecules
mols = np.sort(rgen.randint(0, 20000000, size=100))
# generate sets of normally distributed photons around times of mols
burstsT = [np.sort(rgen.normal(m,20000,rgen.poisson(50)).astype(np.uint64)) for m in mols]
burstsU = [np.sort(rgen.normal(m,20000,rgen.poisson(50)).astype(np.uint64)) for m in mols]
# random exponentially distributed fake nanotimes
nanosT = [rgen.exponential(20, size=m.size).astype(int) for m in burstsT]
nanosU = [rgen.exponential(30, size=m.size).astype(int) for m in burstsU]
# get max index
maxN = max(n.max() for n in chain(nanosT, nanosU))+1
# make fake weights
# TODO: need to make proper orthonormal set
weightT = np.exp(-np.arange(maxN)/20)
weightU = np.exp(-np.arange(maxN)/30)
weightTU = np.vstack([weightT, weightU])

weightsT = tuple(weightT[n] for n in nanosT)
weightsU = tuple(weightU[n] for n in nanosU)
weightsTUT = tuple(weightTU[:,n] for n in nanosT)
weightsTUU = tuple(weightTU[:,n] for n in nanosU)

dataT = (burstsT, nanosT, weightsT)
dataU = (burstsU, nanosU, weightsU)
dataC = tuple(tuple(chain(*p)) for p in product(*((dataT, dataU) for _ in range(2))))

bparam = dataC + tuple(tuple(a[0] for a in d) for d in dataC) + tuple(tuple(np.sort(np.concatenate(a)) for a in d) for d in dataC)


@pytest.fixture(scope='module', params=bparam)
def arrays(request):
    return request.param

@pytest.fixture(scope='module')
def bins():
    return np.logspace(1,4,21).astype(np.uint64)

@pytest.fixture(scope='module', params=list(product([None, False, True],[None, False, True],[None, False, True],[None, False, True])))
def kwargs(request):
    return {key:False for key, val in zip(['minzero', 'normalize', 'bin_width_normalize'], request.param) if  val is not None}

def test_times(arrays, bins, kwargs):
    fcs.correlate(arrays[0], arrays[3], bins, **kwargs)

    
def test_weights(arrays, bins, kwargs):
    fcs.correlate(arrays[0], arrays[3], bins, arrays[2], arrays[5],**kwargs)


def test_weight_nano(arrays, bins, kwargs):
    fcs.correlate(arrays[0], arrays[3], bins, weightsT=weightT, weightsU=weightU, nanosT=arrays[1], nanosU=arrays[4], **kwargs)
