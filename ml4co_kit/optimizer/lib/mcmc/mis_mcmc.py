r"""
MCMC Algorithm for MIS
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
from ml4co_kit.task.graph.mis import MISTask
from ml4co_kit.optimizer.lib.mcmc.pybind11_impl.c_mis_mcmc import pybind11_mis_mcmc_impl


def mis_mcmc_ls(
    task_data: MISTask,
    taus: np.ndarray,
    penalty_coeff: float = 1.001,
    seed: int = 1234,
):
    # Preparation
    edge_index = task_data.edge_index.astype(np.int32)
    nodes_weight = task_data.nodes_weight.astype(np.float64)
    if task_data.sol is None:
        raise ValueError("No solution provided for MIS MCMC.")
    init_sol = task_data.sol
    taus = np.asarray(taus, dtype=np.float64)

    # Call C++ Implementation
    sol = pybind11_mis_mcmc_impl(
        edge_index=edge_index,
        nodes_weight=nodes_weight,
        init_sol=init_sol,
        taus=taus,
        penalty_coeff=penalty_coeff,
        return_trace=False,
        seed=seed,
    )
    sol = np.asarray(sol, dtype=np.int32)

    # Store the solution in the task data
    task_data.from_data(sol=sol, ref=False)