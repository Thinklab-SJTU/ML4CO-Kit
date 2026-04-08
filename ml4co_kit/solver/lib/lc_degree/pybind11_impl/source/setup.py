#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup script for building the PyBind11 LC-degree extension."""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

ext_modules = [
    Pybind11Extension(
        "lc_degree_impl",
        [
            os.path.join(here, "lc_degree.cpp"),
        ],
        include_dirs=[],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="lc_degree_impl",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
