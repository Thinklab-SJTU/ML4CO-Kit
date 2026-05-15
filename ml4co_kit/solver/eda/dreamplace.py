r"""
DreamPlace solver for EDA problems.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import yaml
import pathlib
import importlib.util
from typing import Any, Dict
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.task.eda.edap import EDAPTask
from ml4co_kit.utils.file_utils import download, extract_archive
from ml4co_kit.solver.eda.lib.dreamplace.edap_dreamplace import edap_dreamplace
from ml4co_kit.extension.dreamplace.install_helper import (
    DreamPlaceInstallHelper, DREAMPLACE_THIRDPARTY_PATH
)


class DreamPlaceSolver(SolverBase):
    """
    DreamPlace: https://github.com/limbo018/DREAMPlace
    """

    def __init__(
        self,
        optimizer: OptimizerBase = None,
        hyper_params: Dict[str, Any] = {},
    ):
        # Super Initialization
        super(DreamPlaceSolver, self).__init__(
            SOLVER_TYPE.DREAMPLACE, optimizer=optimizer
        )

        # Extra DreamPlace keys merged on top of YAML (last-wins at top level)
        self.hyper_params = hyper_params

        # Path for DreamPlace YAML files
        self.conf_path = pathlib.Path(__file__).parent / "lib/dreamplace/conf"
    
    def _solve(self, task_data: TaskBase):
        """Solve the task data using DreamPlace solver."""
        # Check if DreamPlace is installed
        self._check_install()
        
        # Solve task_data using DreamPlace
        if task_data.task_type == TASK_TYPE.EDAP:
            params = self.get_params(task_data)
            return edap_dreamplace(task_data=task_data, params=params)
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )

    def get_params(self, task_data: EDAPTask):
        # Get YAML file path
        benchmark_name = task_data.benchmark_name.value
        name = task_data.name
        yaml_path = self.conf_path / benchmark_name / f"{name}.yaml"

        # Check if YAML file exists
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"DreamPlace YAML file not found: {yaml_path}. "
                "This may be because the task does not belong to the supported benchmark datasets. "
                "Please check if the benchmark name and design name are correct."
            )

        # Load YAML file
        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping: {yaml_path}")

        # Merge extra DreamPlace keys
        if self.hyper_params:
            data = {**data, **self.hyper_params}
        data.update({
            "aux_input": task_data.cache["ispd2005_aux"],
            "result_dir": task_data.cache["ispd2005_result_dir"],
        })

        # Create DreamPlaceParams
        from dreamplace.Params import Params as DreamPlaceParams
        params = DreamPlaceParams()
        params.fromJson(data)
        return params

    def install(self, cpu_only: bool = False):
        """Install DreamPlace solver."""
        if not DREAMPLACE_THIRDPARTY_PATH.exists():
            thirdparty_url = (
                "https://huggingface.co/datasets/ML4CO/ML4CO-Kit/"
                "resolve/main/dreamplace_thirdparty.zip"
            )
            download(file_path="dreamplace_thirdparty.zip", url=thirdparty_url)
            extract_archive(
                archive_path="dreamplace_thirdparty.zip",
                extract_path=DREAMPLACE_THIRDPARTY_PATH.as_posix(),
            )

        install_helper = DreamPlaceInstallHelper(cpu_only=cpu_only)
        install_helper.install()

    def _check_install(self):
        self.dreamplace_support = importlib.util.find_spec("dreamplace") is not None
        if not self.dreamplace_support:
            raise ImportError(
                "DreamPlace is not installed. Please install DreamPlace first. "
                "You can quickly install it by calling install()."
            )