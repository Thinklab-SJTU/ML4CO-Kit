r"""
NeuroLKH Solver.
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


import os
import sys
import shutil
import pathlib
from typing import List
from ml4co_kit.utils.file_utils import download
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.neurolkh.tsp_neurolkh import batch_tsp_neurolkh


class NeuroLKHSolver(SolverBase):
    def __init__(
        self,
        lkh_scale: int = 1e6,
        lkh_max_trials: int = 500,
        lkh_path: pathlib.Path = "LKH",
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
        super().__init__(SOLVER_TYPE.NEUROLKH, optimizer=optimizer)
        
        # Initialize Attributes (LKH)
        self.lkh_scale = lkh_scale
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_runs = lkh_runs
        self.lkh_seed = lkh_seed
        self.lkh_special = lkh_special
        
        # Initialize Attributes (NeuroLKH)
        self.neurolkh_device = neurolkh_device
        self.neurolkh_tree_cands_num = neurolkh_tree_cands_num
        self.neurolkh_search_cands_num = neurolkh_search_cands_num
        self.neurolkh_initial_period = neurolkh_initial_period
        self.neurolkh_sparse_factor = neurolkh_sparse_factor
        
        # Check if need download
        if shutil.which(self.lkh_path) is None:
            self.install()  
        
    def _batch_solve(self, batch_task_data: List[TaskBase]):
        """Solve the task data using LKH solver."""
        # Load model
        if batch_task_data[0].task_type == TASK_TYPE.TSP:
            return batch_tsp_neurolkh(
                batch_task_data=batch_task_data,
                lkh_scale=self.lkh_scale,
                lkh_max_trials=self.lkh_max_trials,
                lkh_path=self.lkh_path,
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

    def install(self):
        """Install LKH solver."""
        lkh_url = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz"
        download(file_path="LKH-3.0.13.tgz", url=lkh_url)
        # tar .tgz file
        os.system("tar xvfz LKH-3.0.13.tgz")
        # build LKH
        ori_dir = os.getcwd()
        os.chdir("LKH-3.0.13")
        os.system("make")
        # move LKH to the bin dir
        target_dir = os.path.join(sys.prefix, "bin")
        os.system(f"cp LKH {target_dir}")
        os.chdir(ori_dir)
        # delete .tgz file
        os.remove("LKH-3.0.13.tgz")
        shutil.rmtree("LKH-3.0.13")