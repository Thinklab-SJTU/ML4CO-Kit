r"""
SCIP (Solving Constraint Integer Programs)
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
from ml4co_kit.solver.common.lib.scip.mopo_scip import mopo_scip
from ml4co_kit.solver.common.lib.scip.maxretpo_scip import maxretpo_scip
from ml4co_kit.solver.common.lib.scip.minvarpo_scip import minvarpo_scip


class SCIPSolver(SolverBase):
    """
    SCIP: https://github.com/scipopt/scip
    PySCIPOpt: https://github.com/scipopt/PySCIPOpt
    """
    def __init__(
        self,
        scip_time_limit: float = 10.0,
        soplex_version: str = "8.0.2",
        scip_version: str = "10.0.2",
        optimizer: OptimizerBase = None
    ):
        # Super Initialization
        super(SCIPSolver, self).__init__(
            solver_type=SOLVER_TYPE.SCIP,
            optimizer=optimizer
        )
        
        # Set Attributes
        self.scip_time_limit = scip_time_limit
        self.soplex_version = soplex_version
        self.scip_version = scip_version

        # SOPLEX Path Check
        self.soplex_store_path = sys.prefix
        self.soplex_bin_path = os.path.join(self.soplex_store_path, "bin", "soplex")
        if not os.path.exists(self.soplex_bin_path):
            self.install_soplex()

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

    def install_soplex(self):
        """
        Install SOPLEX.
        SoPlex: https://github.com/soplexopt/soplex
        """
        # Step1: Download SCIP
        soplex_url = (
            f"https://codeload.github.com/scipopt/soplex/"
            f"tar.gz/refs/tags/v{self.soplex_version}"
        )
        download(file_path=f"soplex-{self.soplex_version}.tgz", url=soplex_url)
        
        # Step2: tar .tgz file
        os.system(f"tar xvfz soplex-{self.soplex_version}.tgz")
        os.makedirs(f"soplex-{self.soplex_version}/build", exist_ok=True)
        
        # Step3: build SCIP
        ori_dir = os.getcwd()
        os.chdir(f"soplex-{self.soplex_version}/build")
        cmake_cmd = (
            f"cmake .. "
            f"-DCMAKE_BUILD_TYPE=Release "
            f"-DCMAKE_INSTALL_PREFIX={self.soplex_store_path}"
        )
        os.system(cmake_cmd)
        os.system("make")
        os.system("make install")
        os.chdir(ori_dir)
        
        # Step4: clean up
        os.remove(f"soplex-{self.soplex_version}.tgz")
        shutil.rmtree(f"soplex-{self.soplex_version}")

    def install(self):
        """Install SCIP Solver."""
        # Step1: Download SCIP
        scip_url = (
            f"https://codeload.github.com/scipopt/scip/"
            f"tar.gz/refs/tags/v{self.scip_version}"
        )
        download(file_path=f"SCIP-{self.scip_version}.tgz", url=scip_url)
        
        # Step2: tar .tgz file
        os.system(f"tar xvfz SCIP-{self.scip_version}.tgz")
        os.makedirs(f"scip-{self.scip_version}/build", exist_ok=True)
        
        # Step3: build SCIP
        ori_dir = os.getcwd()
        os.chdir(f"scip-{self.scip_version}/build")
        cmake_cmd = (
            f"cmake .. "
            f"-DAUTOBUILD=on "
            f"-DCMAKE_INSTALL_PREFIX={self.scip_store_path}"
        )
        os.system(cmake_cmd)
        os.system("make")
        os.system("make install")
        os.chdir(ori_dir)
        
        # Step4: clean up
        os.remove(f"SCIP-{self.scip_version}.tgz")
        shutil.rmtree(f"scip-{self.scip_version}")
        msg = (
            f"SCIP Solver installed successfully at {self.scip_store_path}. "
            f"The executable is at {self.scip_bin_path}.\n"
            f"Note: AUTOBUILD=on may skip optional features. For full functionality, "
            f"see https://github.com/scipopt/scip/blob/master/INSTALL.md"
        )
        print(msg)
