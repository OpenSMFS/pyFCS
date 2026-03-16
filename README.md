# pyFCS

[![Tests](https://imgs.xkcd.com/comics/astronomy_status_board.png)](https://xkcd.com/2469/)

## Purpose

Python module with C-level implementation of point-process correlations functions.
These functions were primarily designed for computation of correlation functions
for Fluorescence Correlation Spectroscopy (FCS)[^1] and derivative methods.

Correlations can be computed easily

```python
import numpy as np
import pyFCS as fcs

# Further importats and data processing
# should produce timesT and timesU as
# monotonicaly increasing non-gegative integer arrays

corrl = fcs.correlate(times, times, bins)
```
## Advanced FCS

Beyond standard auto and cross-correlation of FCS data, pyFCS also implements
calculation of correlation with weights for:

1. Fluorescence Lifetime Correlation Spectroscopy (FLCS)[^2]
2. purified FCS (pFCS)[^3]
3. filtered FCS (fFCS)[^4] (the combination of FLCS and pFCS)

All of these will have the same basic form:

```python
import numpy as np
import pyFCS as fcs

# Code gets data for photons and nanotimes,
# Generates the appropriate filter functions
# And provides the times in times
# (in FLCS usually T/U arrays are the same)
# and weights

corrl = fcs.correlate(times, times, bins, weightsT=weights, weightsU=weights)
```

## Implementation

Most of the code is implemented in C for maximal efficiency. The algorithms are all adapted from[^1].
If any issues are encountered that cannot be solved, especially segfaults/kernel crashes, please
open an issue in github, providing the minimal code necessary to reproduce the crash.

[^1]: [Laurence et. al. (2006)](https://doi.org/10.1364/OL.31.000829) Fast, flexible algorithm for calculating photon correlations. *Optics Letters* , 31 (6), 829–831

[^2]: [Bohmer et. al. (2002)](https://doi.org/10.1016/S0009-2614(02)00044-1) Time-resolved fluorescence correlation spectroscopy. *Chem. Phys. Let.* 353 (5-6) 439-445

[^3]: [Laurence et. al.(2007)](https://doi.org/10.1529/biophysj.106.093591) Correlation Spectroscopy of Minor Fluorescent Species: Purification and Distribution Analysis. *Biophysical Journal.* 92, (6), 2184-2198

[^4]: [Felekyan et. al. (2012)](https://doi.org/10.1002/cphc.201100897) Filtered FCS: Species Auto- and Cross-Correlation Functions Highlight Binding and Dynamics in Biomolecules. ChemPhysChem, 13 (4) 1036-1053