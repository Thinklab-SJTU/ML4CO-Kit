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
    from .source.c_astar import c_astar
except:
    c_astar_path = pathlib.Path(__file__).parent / "source"
    ori_dir = os.getcwd()
    os.chdir(c_astar_path)
    os.system("python ./c_astar_setup.py build_ext --inplace")
    os.chdir(ori_dir)
    if os.path.exists(f"{c_astar_path}/build"):
        shutil.rmtree(f"{c_astar_path}/build")
    from .source.c_astar import c_astar