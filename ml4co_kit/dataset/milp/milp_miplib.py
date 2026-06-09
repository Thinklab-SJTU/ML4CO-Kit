r"""
MIPLIB Dataset for MILP.
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
import numpy as np
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.milp.milp import MILPTask
from ml4co_kit.dataset.base import DatasetBase


class MILP_MIPLIBDataset(DatasetBase):
    """
    MIPLIB: https://miplib.zib.de/tag_benchmark.html
    We use the version 2, since the version 1 is deprecated.
    """
    
    def __init__(
        self, 
        precision: Union[np.float32, np.float64] = np.float64,
    ):
        # Super Initialization  
        super(MILP_MIPLIBDataset, self).__init__(
            task_type=TASK_TYPE.MILP,
            dataset_category="milp",
            dataset_name="MIPLIB",
            precision=precision
        )

    def _preprocess(self):
        # Read all ``mps`` files
        problem_files = [
            f for f in os.listdir(self.extracted_save_path / "problems")
            if f.endswith(".mps")
        ]
        problem_files.sort(key=lambda x: x.lower())
        self.cache["problem_files"] = problem_files

    def _load(self, idx) -> MILPTask:
        # Get the miplib files
        mps_file: str = self.cache["problem_files"][idx]
        file_path = self.extracted_save_path / "problems" / mps_file
        sol_file = mps_file.replace(".mps", ".sol")
        sol_path = self.extracted_save_path / "solutions" / sol_file
        if not os.path.exists(sol_path):
            sol_path = None

        # Read the data using ``from_miplib``
        task_data = MILPTask(precision=self.precision)
        task_data.from_miplib(
            file_path=file_path, sol_path=sol_path, ref=True
        )

        # Return task data
        return task_data