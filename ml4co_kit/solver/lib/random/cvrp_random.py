r"""
Random Initialization Algorithm for CVRP
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
from ml4co_kit.task.routing.cvrp import CVRPTask


def _cvrp_random(norm_demands: np.ndarray) -> np.ndarray:
    # Random index
    nodes_num = len(norm_demands)
    index = np.arange(1, nodes_num+1)
    np.random.shuffle(index)

    # Start from depot
    solution = [0]
    load = 0.0

    # Greedy Algorithm for CVRP
    for node in index:
        demand = norm_demands[node - 1]
        if load + demand > 1.0:
            # Need to return to depot
            solution.append(0)
            load = 0.0
        solution.append(node)
        load += demand

    # End at depot
    solution.append(0)
    
    # Convert to numpy array
    return np.array(solution, dtype=np.int32)


def cvrp_random(task_data: CVRPTask):
    # Call ``_cvrp_random`` to get the random tour
    random_tour = _cvrp_random(norm_demands=task_data.norm_demands)

    # Store the random tour in the task_data
    task_data.from_data(sol=random_tour, ref=False)