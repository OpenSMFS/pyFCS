.. currentmodule:: pyFCS

======
Theory
======

Cross-correlation of point processes
------------------------------------

In fluorescence correlation spectroscopy (FCS) the(normalized) [#f1]_ [#f2]_
cross-correlation function (CCF)of two continuous signals :math:`I_1(t)` and
:math:`I_2(t)` is defined as:

.. math::

    G(\tau) = \frac{\langle I_1(t)\; I_2(t) \rangle}
                   {\langle I_1(t)\rangle\langle I_2(t) \rangle}

The auto-correlation function (ACF) is just a special case where :math:`I_1(t) = I_2(t)`.

In actual experiments, signals are not continuous but come from
single-photon detectors that produce a pulse for each photon. These pulses
are usually timestamped with ~1-50 ns resolution. The series of photon
arrival times is used as input for ACF or CCF computations.

In principle, one could bin timestamps to produce a discrete-time signal.
In signal processing, the (non-normalized) cross-correlation of two
real discrete-time signals :math:`\{A_{i}\}` and
:math:`\{B_{i}\}` is defined as :math:`C[m] = \sum_{m=0}^{M} A[i]\ B[i+m]`.
This can be done with the numpy function:
`numpy.correlate <https://docs.scipy.org/doc/numpy/reference/generated/numpy.correlate.html#numpy.correlate>`__
(:math:`\tau = m* bin\:size`).

However, binning timestamps to obtain timetraces would be very inefficient
for FCS analysis where time-lags span many orders of magnitude.
It is much more efficient to compute the cross-correlation function with single
values for ranges (bins) of :math:`\tau` directly from timestamps.
The Laurence algorithm [#f3]_ allows computing cross-correlation
from timestamps on arbitrary bins of time-lags, this is the algorithm implemented
in the :func:`correlate` function of ``pyFCS``.
Computing cross-correlation :math:`C(\tau)` from timestamps is fundamentally
a counting tasks. When considering a single :math:`\tau` value, and given two
timestamps arrays :math:`t` and :math:`u` this is written as

.. math::
    :label: Ctau

    C(\tau) = n(\{(i,j) \ni t_{i} = u_{j} - \tau \})

To be equivalent to a continuous correlation, this counting must be normalized
by the magnitude of the :math:`t` and :math:`u` arrays, resulting in the
following function:

.. math::
    :label: Gtau

    G(\tau) = \frac{n(\{(i, j)\ni t_{i} = u_{i} - \tau\})(T - \tau)}
    {n(\{i \ni t_{i} \le T - \tau\})n(\{j \ni u_{j} \ge \tau\})}

This is the equation given in Laurence et. al. 2006 [#f3]_ paper.

The algorithm however calculates the binned correlation value for the half open
interval :math:`[\tau_{b},\:\tau_{b+1})`, which results in the actual function
:math:`\hat{C}(\tau_{b})` to be defined as follows:

.. math::
    :label: Ctaum

    \hat{C}(\tau_{b}) = n(\{(i, j) \ni \tau_{b} \le u_{j} - t_{i} < \tau_{b+1})\})

:math:`G(\tau)` becomes :math:`G(\tau_{b})` given as:

.. math::
    :label: Gtaub

    \hat{G}(\tau_{b}) = \frac{n(\{(i, j) \ni \tau_{b} \le u_{j} - t_{i} < \tau_{b+1})\})
    (T-\tau_{b+1})}
    {n(\{i \ni t_{i} \le T - \tau_{b+1}\})
    n(\{j \ni u_{j} \ge \tau_{b+1}\})(\tau_{b+1}-\tau_{b})}

The function :func:`correlate` in ``pyFCS`` can calculate both :math:`\hat{C}(\tau_{b})`
and :math:`\hat{G}(\tau_{b}`, through the selection of the ``normalize`` argument.

Newer methods that are extensions of FCS have also been developed, the two primary improvements
are:

1. purified FCS [#f4]_ which correlates multiple paris of arrays
2. Fluorescence Lifetime Correlation Spectroscopy (FLCS) [#f5]_

These two methods can (and often are) be combined, in a technique refrered to as
*filtered* FCS [#f6]_ [#f7]_ .

These are also implemented in the :func:`correlate`.

.. _pfcs:

Purified FCS
------------

Purified FCS modifies :math:`\hat{C}(\tau_{b})` and :math:`\hat{G}(\tau_{b})`
to be a cumulative/average correlation of :math:`K` pairs of :math:`t` and
:math:`u` arrays.

:math:`\hat{C}(\tau_{b})` becomes:

.. math::
    :label: Ctaukb

    \hat{C}(\tau_{b}) = \sum_{k=1}^{K}{n(\{(i, j) \ni \tau_{b} \le u_{k,j} - t_{k,i} < \tau_{b+1})\})}

and :math:`\hat{G}(\tau_{b})` becomes:

.. math::
    :label: Gtaukb

    \hat{G}(\tau_{b}) = \frac{\sum_{K=1}^{K}{n(\{(i, j) \ni \tau_{b} \le u_{k,j} - t_{k,i} < \tau_{b+1}\})}
    (T_{k}-\tau_{b+1})}
    {\sum_{m=1}^{M}{n(\{i \ni t_{k,i} \le T - \tau_{b+1}\})}
    \sum_{m=1}^{M}{n(\{j \ni u_{k,j} \ge \tau_{b+1}\})(\tau_{b+1}-\tau_{b})}}


Purified FCS primarily in single molecule experiments where diffusing fluorophores are at
picomolar concentrations, so that the mean number of fluorophores in the detection volume
is :math:`\ll 1`. In this concentration regime, fluoreophores diffusing into and out of the
detection volume create bursts of photons, which can be isolated from the background with
statistical methods. Then the isolated bursts can be sorted based on various parameters
to select for specific types of species, the selected bursts become the input to purified FCS
in order to obtain the correltation curve(s) of specific species.

It should be noted that if :math:`K = 1` then the above equation is identical
to the previous definition of :math:`\hat{G}(\tau_{m})` .

.. _flcs:

Fluorescence Lifetime Correlation Spectroscopy (FLCS)
-----------------------------------------------------

FLCS leverages the fact that individual data points usually contain more than
just the arrival time that is used in standard FCS correlations. This makes
FLCS not a pure correlation technique, but it also makes it much more powerful
in extracting information.

In FLCS, these additional pieces of information are usually the delay between laser
pulse and photon arrival, (usually called the photon nanotime) but may also be
which detector (spectral, polarization or a combination of the two) at which the
photon arrived.

Whatever the information may be, there should be two or more diffusing species
for which the probabilities of each species in each channel are know. From this
a set of weights arrays that should from an orthonormal basis set that minimizes
the mean square errors between the weigths and raw intensity.

Using these weights to compute the correlation between species :math:`\alpha`
and :math:`\beta`.

.. math::
    :label: Gwtau

    G_{\alpha, \beta}(\tau) = \langle f_{\alpha}I(t) \cdot f_{\beta}I(t+\tau) \rangle

where :math:`f_{\alpha}` and :math:`f_{\beta}` are the weights functions (based on
the channel of the photons at the given time).

For actual calculation this results in the following equation:

.. math::
    :label: Gwtaub

    \hat{G}_{\alpha,\beta}(\tau_{b}) = \frac{\sum_{i,j}^{\tau_{b} \le t_{j}-t_{i} < \tau_{b+1}}{f_{\alpha,i}f_{\beta,j}}(T-\tau_{b})}
    {\sum_{i}^{t_{i} \le T - \tau_{b}}{f_{\alpha,i}}\sum_{j}^{t_{j} \ge \tau_{b}}{f_{\beta,j}}}

filtered FCS
------------

The fitlered FCS and FLCS equations can be combined into the following:

.. math::
    :label: Gwtaukb

    \hat{G}_{\alpha,\beta}(\tau_{b}) = \frac{\{\sum_{m=1}^{M}{[\sum_{i,j}^{\tau_{b} \le t_{j}-t_{i} < \tau_{b+1}}{f_{\alpha,i}f_{\beta,j}}]}\}
    \sum_{m=1}^{M}{(T-\tau_{b})}}
    {\sum_{m=1}^{M}{[\sum_{i}^{t_{i} \le T - \tau_{b}}}{f_{\alpha,i}]}
    \sum_{m=1}^{M}{[\sum_{j}^{t_{j} \ge \tau_{b}}{f_{\beta,j}}}]}

When :math:`M=1`, :eq:`Gwtaukb` reduces to :eq:`Gwtaub`, (in the same way
that :eq:`Gtaukb` reduced to :eq:`Gtaub` ).
:func:`correlate` fundamentally implements either :eq:`Gtaukb` or
:eq:`Gwtaukb` depending on whether a weights function is supplied.

**References**

.. [#f1] Petra Schwille and Elke Haustein,
   `Fluorescence Correlation Spectroscopy  An Introduction to its Concepts and Applications <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.405.2487&rep=rep1&type=pdf>`__
.. [#f2] Haustein, E., Schwille, P. (2003). Ultrasensitive investigations of biological
  systems by fluorescence correlation spectroscopy.
  `*Methods*, *29* (2), 153–166. <https://doi.org/10.1016/S1046-2023(02)00306-7>`__
.. [#f3] Laurence, T. A., Fore, S., Huser, T. (2006). Fast, flexible algorithm for
  calculating photon correlations.  `*Optics Letters* , *31* (6), 829–831. <https://doi.org/10.1364/OL.31.000829>`__
.. [#f4] Laurence, T. A. Correlation Spectroscopy of Minor Fluorescent Species: Signal Purification and Distribution Analysis.
  `*Biophysical Journal* *92* (6)2184-2198 <https://doi.org/10.1529/biophysj.106.093591>`__
.. [#f5] Laurence, T. A., Kwon, Y., Yin, E., Hollars, C. W., Camarero, J. A., Barsky, D., (2007)
  Correlation Spectroscopy of Minor Fluorescent Species: Signal Purification and Distribution Analysis.
  `*Biophysical Journal* *92* (6) 2184-2198 <https://doi.org/10.1529/biophysj.106.093591>`__
.. [#f6] Felekyan, S., Kalinin, S., Veleri, A., Seidel C. A. M., Filtered FCS and species cross correlation function
  `*Proc. SPIE 7183, Multiphoton Microscopy in the Biomedical Sciences* IX, 71830D (13 February 2009) <https://doi.org/10.1117/12.814876>`__
.. [#f7] Felekyan, S., Kalinin, S., Sanabria, H., Valeri, A., Seidel, C. A. M., (2012)
  Filtered FCS: Species Auto- and Cross-Correlation Functions Highlight Binding and Dynamics in Biomolecules
  `*ChemPhysChem* *13* (4) 1036-1053 <https://doi.org/10.1002/cphc.201100897>`__