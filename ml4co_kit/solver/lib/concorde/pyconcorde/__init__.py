r"""
Concorde Solver
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
import shutil
import pathlib


try:
    from .concorde.solve import TSPSolver as TSPConSolver
except:
    concorde_path = pathlib.Path(__file__).parent.parent / "pyconcorde"
    ori_dir = os.getcwd()
    os.chdir(concorde_path)
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{concorde_path}/build"):
        shutil.rmtree(f"{concorde_path}/build")
    from .concorde.solve import TSPSolver as TSPConSolver