r"""
Test MIS MCMC Module.
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
from ml4co_kit import MISTask
from ml4co_kit.extension.mcmc import mis_mcmc


class MISMCMCTester(object):
    """Test cases for MIS MCMC."""
    
    def __init__(self) -> None:
        pass
    
    def test(self):
        # Get MIS Task
        mis_task = MISTask()
        mis_task.from_pickle("test_dataset/graph/mis/task/mis_er-700-800_no-weighted_task.pkl")

        # Test final solution and cost list (Basic)
        final_sol = mis_mcmc(
            task_data=mis_task,
            ref=True,
            penalty_coeff=1.001,
            tau=0.50,
            steps=1000,
            return_type="final_sol",
            return_cost_list=False
        )

        # Test final solution and cost list (Variable Temperature)
        tau_list_1 = np.array([3.0] * 20)
        tau_list_2 = np.linspace(0.10, 0.01, 980)
        tau_list = np.concatenate([tau_list_1, tau_list_2])

        # Test best solution
        best_sol = mis_mcmc(
            task_data=mis_task,
            ref=True,
            penalty_coeff=1.001,
            tau=tau_list,
            steps=1000,
            return_type="best_sol",
            return_cost_list=False
        )  

        # Test mean solution
        mean_sol = mis_mcmc(
            task_data=mis_task,
            ref=True,
            penalty_coeff=1.001,
            tau=tau_list,
            steps=1000,
            return_type="mean_sol",
            return_cost_list=False
        )  

        # Test mean solution
        better_sol_list, cost_list = mis_mcmc(
            task_data=mis_task,
            ref=True,
            penalty_coeff=1.001,
            tau=tau_list,
            steps=1000,
            return_type="better_sol_list",
            return_cost_list=True
        )

        # Test mean solution
        all_sol_list = mis_mcmc(
            task_data=mis_task,
            ref=True,
            penalty_coeff=1.001,
            tau=tau_list,
            steps=1000,
            return_type="all_sol_list",
            return_cost_list=False
        )  