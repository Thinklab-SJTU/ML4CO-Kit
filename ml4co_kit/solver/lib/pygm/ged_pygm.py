r"""
Pygmtools Solver for Graph Matching.
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


import numpy as np
from typing import List
from ml4co_kit.task.qap.ged import GEDTask
from ml4co_kit.extension.pygmtools import PyGMToolsQAPSolver


def ged_pygm(
    task_data: GEDTask, pygm_qap_solver: PyGMToolsQAPSolver
):
    return ged_pygm_batch(
        batch_task_data=[task_data], pygm_qap_solver=pygm_qap_solver
    )


def ged_pygm_batch(
    batch_task_data: List[GEDTask], pygm_qap_solver: PyGMToolsQAPSolver
):
    pass