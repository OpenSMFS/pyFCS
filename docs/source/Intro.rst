============
Introduction
============

The pyFCS module is a fully open source python module, written in C
that is built to support calcualtion of correlations used in
Fluorescence Correlation Spectroscopy, and it's related extensions
and methods.

The primary calcualtion is that of correlation of a point process data,
such as the photon arrival times of data produced by confocal microscopes
equiped with single photon counting detectors.

Installation
------------

.. code-block:: bash

    pip install pyFCS

.. code-block:: bash

    conda install pyFCS

It is also possible to install from source code.

First download the github repository from:

.. code-block:: bash

    git clone https://github.com/harripd/pyFCS


Then the project can be installed by navigating to the project directory,
and running the command

.. code-block:: bash

    pip install .

A C compiler will be required, on Linux and Mac, this will be gcc, supplied out of the box
on Windows, a C compiler will need to be downloaded, most likely with Microsoft Visual Studios.