r"""
OP Wrapper.
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
import pathlib
import numpy as np
from typing import Union, List
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.sat.base import SATTaskBase
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path


class SATWrapperBase(WrapperBase):
    def __init__(
        self, 
        task_type: TASK_TYPE, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(SATWrapperBase, self).__init__(
            task_type=task_type, precision=precision
        )
        self.task_list: List[SATTaskBase] = list()
    
    def _create_task(self) -> SATTaskBase:
        raise NotImplementedError("Subclasses should implement this method.")

    def from_cnf_folder(
        self, 
        cnf_folder_path: pathlib.Path,
        show_time: bool = False                  
    ):
        """Read task data from folder (to support TSPLIB)"""
        # Overwrite task list
        self.task_list: List[SATTaskBase] = list()
        
        # Get file paths and number of instances
        cnf_files = os.listdir(cnf_folder_path)
        cnf_files.sort()
        cnf_files_path = [
            os.path.join(cnf_folder_path, file) 
            for file in cnf_files if file.endswith(".cnf")
        ]
        load_msg = f"Loading data from {cnf_folder_path}"
        for cnf_file_path in tqdm_by_time(
            cnf_files_path, load_msg, show_time
        ):
            sat_task = self._create_task()
            sat_task.from_cnf(file_path=cnf_file_path)
            self.task_list.append(sat_task)
        
    def to_cnf_folder(
        self, 
        cnf_folder_path: pathlib.Path, 
        show_time: bool = False
    ):
        # Write problem of task data (.cnf)
        os.makedirs(cnf_folder_path, exist_ok=True)
        write_msg = f"Writing data to {cnf_folder_path}"
        for task in tqdm_by_time(self.task_list, write_msg, show_time):
            file_name = f"{task.name}.cnf"
            file_path = os.path.join(cnf_folder_path, file_name)
            task.to_cnf(file_path=file_path)
