r"""
MIS MCMC Implementation
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
from enum import Enum
from typing import Union, Any
from ml4co_kit.task.graph.mis import MISTask
from .mis_mcmc_lib import mis_mcmc_enhanced_impl


def mis_mcmc(
    task_data: MISTask,
    ref: bool = True,
    penalty_coeff: float = 1.001,
    tau: Union[float, np.ndarray] = 0.01,
    steps: int = 1000,
    return_type: str = "final_sol",
    return_cost_list: bool = False
) -> Any:
    """
    MIS MCMC Implementation

    return_type: str
        - "final_sol": Final solution after steps
        - "best_sol": Best solution encountered
        - "mean_sol": Average solution
        - "better_sol_list": All solutions with cost >= initial cost (including initial)
        - "all_sol_list": All solutions encountered
    """
    # Preparation
    adj_matrix = task_data.to_adj_matrix()
    weights = task_data.nodes_weight
    if ref:
        init_sol = task_data.ref_sol
    else:
        init_sol = task_data.sol

    # Ensure correct dtypes
    adj_matrix = adj_matrix.astype(np.int32)
    weights = weights.astype(np.float64)
    init_sol = init_sol.astype(np.int32)

    # Call C++ implementation (init_cost is calculated automatically in C++)
    result = mis_mcmc_enhanced_impl(
        adj_matrix, weights, init_sol, penalty_coeff, tau, 
        steps, return_type, return_cost_list
    )

    return result