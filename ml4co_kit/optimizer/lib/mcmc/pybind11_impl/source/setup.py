#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))

ext_modules = [
    Pybind11Extension(
        "mcmc_impl",
        [
            os.path.join(here, "mcmc.cpp"),
            os.path.join(here, "bindings.cpp"),
        ],
        include_dirs=[here],
        language="c++",
        cxx_std=11,
    ),
]

setup(
    name="mcmc_impl",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
