r"""
Greedy Algorithm for UNSAT-C
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
from ml4co_kit.task.sat.unsatc import USATCTask


def unsatc_greedy(task_data: USATCTask):
    var_freq = np.zeros(task_data.vars_num, dtype=np.int32)
    
    for clause in task_data.clauses:
        for lit in clause:
            var_idx = abs(lit) - 1
            if var_idx < task_data.vars_num:
                var_freq[var_idx] += 1
    
    if "heatmap" in task_data.cache:
        heatmap = task_data.cache["heatmap"]
        combined_score = var_freq * (1 + heatmap)
    else:
        combined_score = var_freq
    
    threshold = np.percentile(combined_score, 70)
    sol = (combined_score >= threshold).astype(bool)
    
    task_data.sol = sol
