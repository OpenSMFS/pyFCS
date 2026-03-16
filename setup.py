#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:07:39 2024

@author: paul
"""

from setuptools import setup, Extension
import numpy as np

extension = [Extension("pyFCS", sources=["pyFCS/cFCS.c","pyFCS/correlate.c", 
                                         "pyFCS/correlate_thread.c", "pyFCS/correlateinterface.c"], 
                       CFLAGS="-g", include_dirs=[np.get_include()])]
# extension = [Extension("helloworld", ["hwrd.c",])]

setup(name='pyFCS',
      version='0.1.1',
      description='Pure C python module for calculating Fluorescence Correlation curves .',
      ext_modules=extension
      )
