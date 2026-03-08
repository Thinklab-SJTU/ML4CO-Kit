r"""
FEM (Free Energy Minimization) Library.

This module contains problem-specific implementations of the FEM algorithm.
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


from .mcut_fem import mcut_fem
from .mis_fem import mis_fem
from .mvc_fem import mvc_fem
from .mcl_fem import mcl_fem

__all__ = ['mcut_fem', 'mis_fem', 'mvc_fem', 'mcl_fem']
