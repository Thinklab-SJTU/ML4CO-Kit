#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))

ext_modules = [
    Pybind11Extension(
        "cvrp_nearest_impl",
        [
            os.path.join(here, "cvrp_nearest.cpp"),
        ],
        include_dirs=[],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="cvrp_nearest_impl",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
