r"""
SCIP Solver for Portfolio Optimization Problems.
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


import os
import sys
import shutil
from ml4co_kit.utils.file_utils import download
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.solver.lib.scip.mopo_scip import mopo_scip
from ml4co_kit.solver.lib.scip.maxretpo_scip import maxretpo_scip
from ml4co_kit.solver.lib.scip.minvarpo_scip import minvarpo_scip


class SCIPSolver(SolverBase):
    def __init__(
        self,
        scip_time_limit: float = 10.0,
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(SCIPSolver, self).__init__(
            solver_type=SOLVER_TYPE.SCIP,
            optimizer=optimizer
        )
        
        # Set Attributes
        self.scip_time_limit = scip_time_limit

        # SCIP Path Check
        self.scip_store_path = sys.prefix
        self.scip_bin_path = os.path.join(self.scip_store_path, "bin", "scip")
        if not os.path.exists(self.scip_bin_path):
            self.install()

    def _solve(self, task_data: TaskBase):
        """Solve the task data using SCIP Solver."""
        if task_data.task_type == TASK_TYPE.MAXRETPO:
            return maxretpo_scip(
                task_data=task_data,
                scip_time_limit=self.scip_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MINVARPO:
            return minvarpo_scip(
                task_data=task_data,
                scip_time_limit=self.scip_time_limit
            )
        elif task_data.task_type == TASK_TYPE.MOPO:
            return mopo_scip(
                task_data=task_data,
                scip_time_limit=self.scip_time_limit
            )
        else:
            raise ValueError(
                f"SCIP Solver does not support task type: {task_data.task_type}. "
                f"Supported types: MaxRetPO, MinVarPO, MOPO"
            )

    def install(self):
        """Install SCIP Solver."""
        # Step1: Download SCIP
        scip_url = "https://codeload.github.com/scipopt/scip/tar.gz/refs/tags/v923"
        download(file_path="SCIP-9.2.3.tgz", url=scip_url)
        
        # Step2: tar .tgz file
        os.system("tar xvfz SCIP-9.2.3.tgz")
        os.makedirs("scip-923/build", exist_ok=True)
        
        # Step3: build SCIP
        ori_dir = os.getcwd()
        os.chdir("scip-923/build")
        os.system(f"cmake .. -DAUTOBUILD=on -DCMAKE_INSTALL_PREFIX={self.scip_store_path}")
        os.system("make")
        os.system("make install")
        os.chdir(ori_dir)
        
        # Step4: clean up
        os.remove("SCIP-9.2.3.tgz")
        shutil.rmtree("scip-923")
        msg = (
            f"SCIP Solver installed successfully at {self.scip_store_path}. "
            f"The executable is at {self.scip_bin_path}.\n"
            f"Note: AUTOBUILD=on may skip optional features. For full functionality, "
            f"see https://github.com/scipopt/scip/blob/master/INSTALL.md"
        )
        print(msg)