r"""
Utilities for implementation selection.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from enum import Enum


class IMPL_TYPE(str, Enum):
    """Define the implementation types as an enumeration."""
    
    AUTO = "auto"
    CTYPES = "ctypes"
    CYTHON = "cython"
    NUMPY = "numpy"
    PYBIND11 = "pybind11"
    TORCH = "torch"