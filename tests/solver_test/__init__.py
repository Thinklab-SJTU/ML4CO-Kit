r"""
Solver Test Module.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from .base import SolverBase
from .concorde import ConcordeSolverTester
from .gp_degree import GpDegreeSolverTester
from .greedy import GreedySolverTester
from .lkh import LKHSolverTester
from .lc_degree import LcDegreeSolverTester
from .insertion import InsertionSolverTester
from .rlsa import RLSASolverTester
from .hgs import HGSSolverTester