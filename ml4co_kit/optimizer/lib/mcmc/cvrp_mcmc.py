r"""
MCMC Algorithm for CVRP
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
from ml4co_kit.optimizer.lib.cvrp_ls.cvrp_ls import cvrp_ls
from ml4co_kit.optimizer.lib.mcmc.pybind11_impl.c_cvrp_mcmc import pybind11_cvrp_mcmc_impl


def cvrp_mcmc_ls(
    task_data: CVRPTask, 
    taus: np.ndarray,
    penalty_coeff: float = 1.001,
    seed: int = 1234,
):
    # Preparation
    coords = task_data.coords.astype(np.float64)
    norm_demands = task_data.norm_demands.astype(np.float64)
    if task_data.sol is None:
        raise ValueError("No solution provided for CVRP MCMC.")
    init_sol = task_data.sol
    taus = np.asarray(taus, dtype=np.float64)

    # Call C++ Implementation
    sol = pybind11_cvrp_mcmc_impl(
        coords=coords,
        norm_demands=norm_demands,
        init_sol=init_sol,
        taus=taus,
        penalty_coeff=penalty_coeff,
        return_trace=False,
        seed=seed,
    )
    sol = np.asarray(sol, dtype=np.int32)
    
    # Perform local search to repair
    task_data.from_data(sol=sol, ref=False)
    cvrp_ls(task_data=task_data, seed=seed)

    # Final Check
    if not task_data.check_constraints(task_data.sol):
        task_data.from_data(sol=init_sol, ref=False)
        cvrp_ls(task_data=task_data, seed=seed)
