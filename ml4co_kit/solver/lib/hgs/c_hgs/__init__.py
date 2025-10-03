r"""
C++ Solver for HGS
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


import os
import pathlib


# C++ HGS Solver Path
HGS_BASE_PATH = pathlib.Path(__file__).parent
HGS_SOLVER_PATH = HGS_BASE_PATH / "cvrp_hgs_solver"


# Determining whether the solvers have been built
if not os.path.exists(HGS_SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(HGS_BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)  