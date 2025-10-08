r"""
LKH Solver.
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
from ml4co_kit.utils.file_utils import download
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.lib.lkh.tsp_lkh import tsp_lkh
from ml4co_kit.solver.lib.lkh.atsp_lkh import atsp_lkh
from ml4co_kit.solver.lib.lkh.cvrp_lkh import cvrp_lkh
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class LKHSolver(SolverBase):
    def __init__(
        self,
        lkh_scale: int = 1e6,
        lkh_max_trials: int = 500,
        lkh_path: pathlib.Path = "LKH",
        lkh_runs: int = 1,
        lkh_seed: int = 1234,
        lkh_special: bool = False,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super().__init__(SOLVER_TYPE.LKH, optimizer=optimizer)

        # Initialize Attributes
        self.lkh_scale = lkh_scale
        self.lkh_max_trials = lkh_max_trials
        self.lkh_path = lkh_path
        self.lkh_runs = lkh_runs
        self.lkh_seed = lkh_seed
        self.lkh_special = lkh_special
        
        # Check if need download
        if shutil.which(self.lkh_path) is None:
            self.install()
            
    def _solve(self, task_data: TaskBase):
        """Solve the task data using LKH solver."""
        if task_data.task_type == TASK_TYPE.TSP:
            return tsp_lkh(
                task_data=task_data,
                lkh_scale=self.lkh_scale,
                lkh_max_trials=self.lkh_max_trials,
                lkh_path=self.lkh_path,
                lkh_runs=self.lkh_runs,
                lkh_seed=self.lkh_seed,
                lkh_special=self.lkh_special
            )
        elif task_data.task_type == TASK_TYPE.ATSP:
            return atsp_lkh(
                task_data=task_data,
                lkh_scale=self.lkh_scale,
                lkh_max_trials=self.lkh_max_trials,
                lkh_path=self.lkh_path,
                lkh_runs=self.lkh_runs,
                lkh_seed=self.lkh_seed,
                lkh_special=self.lkh_special
            )
        elif task_data.task_type == TASK_TYPE.CVRP:
            return cvrp_lkh(
                task_data=task_data,
                lkh_scale=self.lkh_scale,
                lkh_max_trials=self.lkh_max_trials,
                lkh_path=self.lkh_path,
                lkh_runs=self.lkh_runs,
                lkh_seed=self.lkh_seed,
                lkh_special=self.lkh_special
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
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