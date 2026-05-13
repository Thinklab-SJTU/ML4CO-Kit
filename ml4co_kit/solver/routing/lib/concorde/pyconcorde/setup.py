""" setup command to build Concorde Cython wrappers.

By default, the setup script will download the QSOpt linear solver, and
download and compile Concorde. If you have either one of these packages
already installed, you can use the installed version by setting the
following environment variables:

QSOPT_DIR: should point to a folder containing qsopt.a and qsopt.h
CONCORDE_DIR: contains concorde.a and concorde.h

Note that for the build process to work correctly, you should either
not set these variables (and rely on the downloaded Concorde) or set
both of them. Setting only one will not work as intended.
"""


import os
import shutil
import platform
import subprocess
import numpy as np
from Cython.Build import cythonize
from os.path import exists, join as pjoin
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def download_concorde_qsopt():
    # Make sure the build directories exist
    os.makedirs("build", exist_ok=True)

    # Get system and machine type
    system_name = platform.system()
    machine_type = platform.machine()
    
    # copy QSOPT files from data/{system_name}_{machine_type} to data directory
    qsopt_a_path = pjoin("data", f"{system_name}_{machine_type}", "qsopt.a")
    qsopt_h_path = pjoin("data", f"{system_name}_{machine_type}", "qsopt.h")
    if not os.path.exists(qsopt_a_path):
        raise ValueError(f"QSOPT files for {system_name}_{machine_type} are missing")
    shutil.copy(qsopt_a_path, pjoin("data", "qsopt.a"))
    shutil.copy(qsopt_h_path, pjoin("data", "qsopt.h"))

    # Copy concorde codes from src to build
    concorde_src_path = "src"
    concorde_build_path = pjoin("build", "concorde")
    shutil.copytree(concorde_src_path, concorde_build_path)


def _run(cmd, cwd):
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def build_concorde():
    if not exists("data/concorde.h") or not exists("data/concorde.a"):
        print("building concorde")

        cflags = "-fPIC -O2 -g"

        if platform.system().startswith("Darwin"):
            flags = "--host=darwin"
        else:
            flags = ""

        datadir = os.path.abspath("data")
        cwd = (
            'CFLAGS="{cflags}" ./configure --prefix {data} '
            "--with-qsopt={data} {flags}"
        ).format(cflags=cflags, data=datadir, flags=flags)

        _run(cwd, "build/concorde")
        _run("make", "build/concorde")

        shutil.copyfile("build/concorde/concorde.a", "data/concorde.a")
        shutil.copyfile("build/concorde/concorde.h", "data/concorde.h")


class build_ext(_build_ext, object):
    """Build command that downloads and installs Concorde, if not found."""

    def run(self):
        if not self.has_external_concorde:
            download_concorde_qsopt()
            build_concorde()
        else:
            print("Using external Concorde/QSOpt")

        super(build_ext, self).run()

    @property
    def has_external_concorde(self):
        qsopt_dir = os.environ.get("QSOPT_DIR")
        concorde_dir = os.environ.get("CONCORDE_DIR")
        return bool(qsopt_dir) and bool(concorde_dir)


class ConcordeExtension(Extension, object):
    """Extension that sets Concorde/QSOpt lib/include args."""

    def __init__(self, *args, **kwargs):
        super(ConcordeExtension, self).__init__(*args, **kwargs)
        qsopt_dir = os.environ.get("QSOPT_DIR", "data")
        concorde_dir = os.environ.get("CONCORDE_DIR", "data")
        self.include_dirs.append(concorde_dir)
        self.extra_objects.extend(
            [pjoin(qsopt_dir, "qsopt.a"), pjoin(concorde_dir, "concorde.a")]
        )


setup(
    name="pyconcorde",
    ext_modules=cythonize(
        [
            ConcordeExtension(
                "concorde._concorde",
                sources=["concorde/_concorde.pyx"],
                include_dirs=[np.get_include()],
            )
        ]
    ),
    version="0.1.0",
    install_requires=[
        "cython>=0.22.0",
        "numpy>=1.21.0",
        "tsplib95",
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    license="BSD",
    author="Joris Vankerschaver",
    author_email="joris.vankerschaver@gmail.com",
    url="https://github.com/jvkersch/pyconcorde",
    description="Cython wrappers for the Concorde TSP library",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
    ],
    cmdclass={
        "build_ext": build_ext,
    },
)
