#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for building the PyBind11 MIS MCMC extension.
"""

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


# Get the directory containing this setup.py
here = os.path.abspath(os.path.dirname(__file__))


# Define the extension module
ext_modules = [
    Pybind11Extension(
        "mis_mcmc_lib",
        [
            os.path.join(here, "mis.cpp"),
            os.path.join(here, "bindings.cpp"),
        ],
        include_dirs=[here],
        language="c++",
        cxx_std=11,
    ),
]

setup(
    name="mis_mcmc_lib",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

