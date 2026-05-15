r"""
LKH (Lin-Kernighan-Helsgaun)
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


from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.utils.file_utils import download, extract_archive
from ml4co_kit.extension.dreamplace.install_helper import (
    DreamPlaceIntallHelper, DREAMPLACE_THIRDPARTY_PATH
)


class DreamPlaceSolver(SolverBase):
    """
    DreamPlace: https://github.com/limbo018/DREAMPlace
    """
    def __init__(
        self,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(DreamPlaceSolver, self).__init__(SOLVER_TYPE.DREAMPLACE, optimizer=optimizer)

    def _solve(self, task_data: TaskBase):
        """Solve the task data using DreamPlace solver."""
        if task_data.task_type == TASK_TYPE.EDAP:
            return edap_dreamplace(
                task_data=task_data,
                solver_name="lingeling",
                solver_args={"phase": True}
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )

    def install(self, cpu_only: bool = False):
        """Install DreamPlace solver."""
        # Step1: Download thirdparty
        if not DREAMPLACE_THIRDPARTY_PATH.exists():
            thirdparty_url = "https://huggingface.co/datasets/ML4CO/ML4CO-Kit/resolve/main/dreamplace_thirdparty.zip"
            download(file_path="dreamplace_thirdparty.zip", url=thirdparty_url)
            extract_archive(
                archive_path="dreamplace_thirdparty.zip", 
                extract_path=DREAMPLACE_THIRDPARTY_PATH.as_posix()
            )
        
        # Step2: Get Install Helper
        install_helper = DreamPlaceIntallHelper(cpu_only=cpu_only)

        # Step3: Install DreamPlace
        install_helper.install()