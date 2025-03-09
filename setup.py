#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import re
from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from wheel._bdist_wheel import get_platform, get_abi_tag, tags


def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


# Package meta-data.
NAME = "ml4co-kit"
PACKAGE_NAME = "ml4co_kit"
DESCRIPTION = "ml4co-kit provides convenient dataset generators for the combinatorial optimization problem"
URL = "https://github.com/Thinklab-SJTU/ML4CO-Kit"
AUTHOR = get_property("__author__", PACKAGE_NAME)
VERSION = get_property("__version__", PACKAGE_NAME)
REQUIRED = [
    "numpy>=1.24.4",
    "networkx>=2.8.8",
    "tqdm>=4.66.3",
    "pulp>=2.8.0",
    "pandas>=2.0.0",
    "scipy>=1.10.1",
    "aiohttp>=3.10.11",
    "requests>=2.32.0",
    "async_timeout>=4.0.3",
    "pyvrp>=0.6.3",
    "cython>=3.0.8",
    "gurobipy>=11.0.3",
    "scikit-learn>=1.3.0"    
]

EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class BdistWheelCommand(_bdist_wheel):
    def run(self):
        os.system("python publish/compile_solvers.py")
        super().run()

    def get_tag(self):
        if self.plat_name and not self.plat_name.startswith("macosx"):
            plat_name = self.plat_name
        else:
            plat_name = get_platform(self.bdist_dir)
        if plat_name in ("linux-x86_64", "linux_x86_64"):
            plat_name = "manylinux2014_x86_64"
        plat_name = (
            plat_name.lower().replace("-", "_").replace(".", "_").replace(" ", "_")
        )
        impl_name = tags.interpreter_name()
        impl_ver = tags.interpreter_version()
        impl = impl_name + impl_ver
        abi_tag = str(get_abi_tag()).lower()
        tag = (impl, abi_tag, plat_name)
        return tag


setup(
    name=PACKAGE_NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    url=URL,
    packages=find_packages(),
    package_data={PACKAGE_NAME: ["**"], NAME: ["**"], "docs": ["**"]},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Mulan PSL v2",
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: Mulan Permissive Software License v2 (MulanPSL-2.0)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: Unix",
        "Environment :: Console",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    # $ setup.py publish support.
    cmdclass={
        "bdist_wheel": BdistWheelCommand,
    },
)
