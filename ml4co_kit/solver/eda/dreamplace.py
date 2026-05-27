r"""
DreamPlace solver for EDA problems.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import yaml
import pathlib
import importlib.util
from typing import Any, Dict
from ml4co_kit.task.eda.edap import EDAPTask
from ml4co_kit.task.eda.base import EDA_BENCH
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE
from ml4co_kit.utils.file_utils import download, extract_archive
from ml4co_kit.extension.dreamplace.install_helper import (
    DreamPlaceInstallHelper, DreamPlaceBuildEnvChecker,
    DreamPlacePrebuilt, DREAMPLACE_THIRDPARTY_PATH,
)
from .lib.dreamplace.edap_dreamplace import edap_dreamplace


class DreamPlaceSolver(SolverBase):
    """
    DreamPlace: https://github.com/limbo018/DREAMPlace
    Current Version: 37214b40fe3837cc7d392c7d6092ccd6ff04a02c
    Last Update: 2026-05-26
    @inproceedings{
        lin2019dreamplace,
        title={Dreamplace: Deep learning toolkit-enabled gpu acceleration for modern vlsi placement},
        author={Lin, Yibo and Dhar, Shounak and Li, Wuxi and Ren, Haoxing and Khailany, Brucek and Pan, David Z},
        booktitle={Proceedings of the 56th Annual Design Automation Conference 2019},
        pages={1--6},
        year={2019}
    }
    @inproceedings{
        lin2020dreamplace,
        title={DREAMPlace 2.0: Open-source GPU-accelerated global and detailed placement for large-scale VLSI designs},
        author={Lin, Yibo and Pan, David Z and Ren, Haoxing and Khailany, Brucek},
        booktitle={2020 China Semiconductor Technology International Conference (CSTIC)},
        pages={1--4},
        year={2020},
        organization={IEEE}
    }
    @inproceedings{
        gu2020dreamplace,
        title={DREAMPlace 3.0: Multi-electrostatics based robust VLSI placement with region constraints},
        author={Gu, Jiaqi and Jiang, Zixuan and Lin, Yibo and Pan, David Z},
        booktitle={Proceedings of the 39th International Conference on Computer-Aided Design},
        pages={1--9},
        year={2020}
    }
    @inproceedings{
        liao2022dreamplace,
        title={DREAMPlace 4.0: Timing-driven global placement with momentum-based net weighting},
        author={Liao, Peiyu and Liu, Siting and Chen, Zhitang and Lv, Wenlong and Lin, Yibo and Yu, Bei},
        booktitle={2022 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
        pages={939--944},
        year={2022},
        organization={IEEE}
    }
    """
    def __init__(
        self,
        device: str = "cpu",
        optimizer: OptimizerBase = None,
        hyper_params: Dict[str, Any] = {},
    ):
        # Super Initialization
        super(DreamPlaceSolver, self).__init__(
            SOLVER_TYPE.DREAMPLACE, optimizer=optimizer
        )

        # Set device
        self.device = device

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

        # Device
        gpu = 0 if self.device == "cpu" else 1

        # Merge extra DreamPlace keys
        if self.hyper_params:
            data = {**data, **self.hyper_params}
        
        # Update data according to different benchmark
        bn = task_data.benchmark_name
        if bn in [EDA_BENCH.ISPD2005, EDA_BENCH.ISPD2005FREE, EDA_BENCH.MMS]:
            data.update({
                "aux_input": task_data.cache["aux"],
                "result_dir": task_data.cache["result_dir"],
                "gpu": gpu,
            })
        else:
            raise NotImplementedError(
                f"{bn.value} is not supported for DreamPlace solver. "
            )

        # Create DreamPlaceParams
        from dreamplace.Params import Params as DreamPlaceParams
        params = DreamPlaceParams()
        params.fromJson(data)
        return params

    def install(self, cpu_only: bool = False, using_prebuilt: bool = False):
        """
        Install DreamPlace solver.
        """
        # Create DreamPlaceInstallHelper
        install_helper = DreamPlaceInstallHelper(cpu_only=cpu_only)

        # Using Prebuilt DreamPlace
        if using_prebuilt:
            if not cpu_only:
                raise ValueError(
                    "Prebuilt DreamPlace packages are CPU-only. "
                    "Call install(cpu_only=True, using_prebuilt=True)."
                )
            else:
                print(
                    "Note: We do not recommend using the prebuilt DreamPlace package. "
                    "The prebuilt files are mainly provided for our development workflow "
                    "on GitHub Actions, to avoid unnecessary repeated compilation. "
                    "To use prebuilt packages, your environment must match exactly with "
                    "the build environment on GitHub. In practice, this is difficult to "
                    "guarantee, so compiling through DreamPlaceInstallHelper is preferred."
                )
                return DreamPlacePrebuilt(install_helper.final_path).install()

        # Step1: Check DreamPlace build environment
        DreamPlaceBuildEnvChecker.check()

        # Step2: Download and extract DreamPlace thirdparty
        if not DREAMPLACE_THIRDPARTY_PATH.exists():
            thirdparty_url = (
                "https://huggingface.co/datasets/ML4CO/ML4CO-Kit/"
                "resolve/main/dreamplace/dreamplace_thirdparty.zip"
            )
            download(file_path="dreamplace_thirdparty.zip", url=thirdparty_url)
            extract_archive(
                archive_path="dreamplace_thirdparty.zip",
                extract_path=DREAMPLACE_THIRDPARTY_PATH.as_posix(),
            )
            os.remove("dreamplace_thirdparty.zip")

        # Step3: Compile DreamPlace
        install_helper.install()

    def _check_install(self):
        self.dreamplace_support = importlib.util.find_spec("dreamplace") is not None
        if not self.dreamplace_support:
            raise ImportError(
                "DreamPlace is not installed. Please install DreamPlace first. "
                "You can quickly install it by calling install()."
            )