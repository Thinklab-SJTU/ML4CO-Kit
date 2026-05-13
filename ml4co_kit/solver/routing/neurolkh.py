r"""
NeuroLKH
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from typing import List
from ml4co_kit.solver.base import SOLVER_TYPE
from ml4co_kit.solver.routing.lkh import LKHSolver
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.routing.lib.neurolkh.tsp_neurolkh import batch_tsp_neurolkh


class NeuroLKHSolver(LKHSolver):
    """
    NeuroLKH: https://github.com/liangxinedu/NeuroLKH
    @article{
        xin2021neurolkh,
        title={Neurolkh: Combining deep learning model with lin-kernighan-helsgaun \
            heuristic for solving the traveling salesman problem},
        author={Xin, Liang and Song, Wen and Cao, Zhiguang and Zhang, Jie},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        pages={7472--7483},
        year={2021}
    }
    """
    def __init__(
        self,
        lkh_scale: int = 1e6,
        lkh_max_trials: int = 500,
        lkh_runs: int = 1,
        lkh_seed: int = 1234,
        lkh_special: bool = False, 
        neurolkh_device: str = "cpu",
        neurolkh_tree_cands_num: int = 10,
        neurolkh_search_cands_num: int = 5,
        neurolkh_initial_period: int = 15,
        neurolkh_sparse_factor: int = 20,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(NeuroLKHSolver, self).__init__(
            lkh_scale=lkh_scale,
            lkh_max_trials=lkh_max_trials,
            lkh_runs=lkh_runs,
            lkh_seed=lkh_seed,
            lkh_special=lkh_special,
            optimizer=optimizer
        )
        
        # Initialize Attributes (NeuroLKH)
        self.solver_type = SOLVER_TYPE.NEUROLKH
        self.neurolkh_device = neurolkh_device
        self.neurolkh_tree_cands_num = neurolkh_tree_cands_num
        self.neurolkh_search_cands_num = neurolkh_search_cands_num
        self.neurolkh_initial_period = neurolkh_initial_period
        self.neurolkh_sparse_factor = neurolkh_sparse_factor
        
    def _batch_solve(self, batch_task_data: List[TaskBase]):
        """Solve the task data using LKH solver."""
        # Load model
        if batch_task_data[0].task_type == TASK_TYPE.TSP:
            return batch_tsp_neurolkh(
                batch_task_data=batch_task_data,
                lkh_scale=self.lkh_scale,
                lkh_max_trials=self.lkh_max_trials,
                lkh_path=self.lkh_bin_path,
                lkh_runs=self.lkh_runs,
                lkh_seed=self.lkh_seed,
                lkh_special=self.lkh_special,
                neurolkh_device=self.neurolkh_device,
                neurolkh_tree_cands_num=self.neurolkh_tree_cands_num,
                neurolkh_search_cands_num=self.neurolkh_search_cands_num,
                neurolkh_initial_period=self.neurolkh_initial_period,
                neurolkh_sparse_factor=self.neurolkh_sparse_factor,
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {batch_task_data[0].task_type}."
            )