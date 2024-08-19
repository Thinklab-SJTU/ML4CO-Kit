import numpy as np
from glob import glob
from setuptools import setup, Extension
from Cython.Build import cythonize


setup(
    name='cython_tsp_greedy function',
    ext_modules=cythonize(
        Extension(
            'cython_tsp_greedy',
            glob('*.pyx'),
            include_dirs=[np.get_include(),"."],
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-std=c++11"],
        ),
        language_level = "3",
    ),
    zip_safe=False,
)
