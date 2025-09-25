r"""
Greedy Algorithm for MCUT
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
from ml4co_kit.task.graph.mcut import MCutTask


def mcut_greedy(task_data: MCutTask):
    # Greedy Algorithm for MCUT
    heatmap: np.ndarray = task_data.cache["heatmap"]
    sol = (heatmap > 0.5).astype(np.int32)
    
    # Store the tour in the task_data
    task_data.from_data(sol=sol, ref=False)