r"""
KaMIS Algorithm for MIS
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
from chszlablib import IndependenceProblems
from ml4co_kit.task.graph.mis import MISTask


def mis_kamis(
    task_data: MISTask,
    kamis_time_limit: float,
    unweighted_solver_type: str = "redumis",
    weighted_solver_type: str = "mmwis",
    weighted_scale: float = 1e5,
    kamis_seed: int = 1234
):
    # Convert the task data to a chszlablib graph
    chszlablib_graph = task_data.to_chszlablib(weighted_scale)
    
    # Solve the MIS problem
    if task_data.node_weighted:
        if weighted_solver_type == "mmwis":
            result = IndependenceProblems.mmwis(
                g=chszlablib_graph, 
                time_limit=kamis_time_limit, 
                seed=kamis_seed
            )
        elif weighted_solver_type == "branch_reduce":
            result = IndependenceProblems.branch_reduce(
                g=chszlablib_graph, 
                time_limit=kamis_time_limit,
                seed=kamis_seed
            )
        else:
            raise ValueError(f"Invalid weighted solver type: {weighted_solver_type}")
    else:
        if unweighted_solver_type == "redumis":
            result = IndependenceProblems.redumis(
                g=chszlablib_graph, 
                time_limit=kamis_time_limit,
                seed=kamis_seed
            )
        elif unweighted_solver_type == "online_mis":
            result = IndependenceProblems.online_mis(
                g=chszlablib_graph, time_limit=kamis_time_limit)
        else:
            raise ValueError(f"Invalid unweighted solver type: {unweighted_solver_type}")

    # Convert the result to a solution vector
    sol = np.zeros(task_data.nodes_num, dtype=np.int32)
    sol[np.asarray(result.vertices, dtype=np.int32)] = 1
    task_data.from_data(sol=sol)