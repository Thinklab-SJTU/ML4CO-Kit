r"""
C++ Solver for PCTSP ILS
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


# C++ PCTSP ILS Solver Path
BASE_PATH = pathlib.Path(__file__).parent
C_PCTSP_ILS_SOLVER_PATH = pathlib.Path(__file__).parent / "pctsp_ils"


# Determining whether the solvers have been built
if not os.path.exists(C_PCTSP_ILS_SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)