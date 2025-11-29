#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for building the PyBind11 TSP 2-opt extension.
"""

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


# Get the directory containing this setup.py
here = os.path.abspath(os.path.dirname(__file__))


# Define the extension module
ext_modules = [
    Pybind11Extension(
        "tsp_2opt_impl",
        [
            os.path.join(here, "tsp_2opt.cpp"),
        ],
        include_dirs=[],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="tsp_2opt_impl",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

