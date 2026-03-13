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


import pathlib
import numpy as np
from typing import Union, List
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.sat.sata import SATATask
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.wrapper.sat.base import SATWrapperBase
from ml4co_kit.utils.file_utils import check_file_path


class SATAWrapper(SATWrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(SATAWrapper, self).__init__(
            task_type=TASK_TYPE.SATA, precision=precision
        )
        self.task_list: List[SATATask] = list()

    def _create_task(self) -> SATATask:
        return SATATask(precision=self.precision)

    def from_txt(
        self, 
        file_path: pathlib.Path,
        ref: bool = False,
        overwrite: bool = True,
        normalize: bool = False,
        show_time: bool = False
    ):
        """Read task data from ``.txt`` file"""
        # Overwrite task list if overwrite is True
        if overwrite:
            self.task_list: List[SATATask] = list()
        
        # Read task data from ``.txt`` file
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                split_first = line.split(" clauses ")
                vars_num = int(split_first[0])
                split_second = split_first[1].split(" label ")
                clauses_split = split_second[0].split(" 0 ")
                clauses = [clause.split(" ") for clause in clauses_split]
                clauses = [[int(literal) for literal in clause] for clause in clauses]
                clauses[-1] = clauses[-1][:-1] # Remove the last empty clause
                sol = split_second[1]
                sol = sol.split(" ")
                sol = np.array([int(t) for t in sol]).astype(np.bool_)
                
                # Create a new task and add it to ``self.task_list``
                if overwrite:
                    sata_task = SATATask(precision=self.precision)
                else:
                    sata_task = self.task_list[idx]
                sata_task.from_data(
                    clauses=clauses, vars_num=vars_num, sol=sol, ref=ref
                )
                if overwrite:
                    self.task_list.append(sata_task)
    
    def to_txt(
        self, file_path: pathlib.Path, show_time: bool = False, mode: str = "w"
    ):
        """Write task data to ``.txt`` file"""
        # Check file path
        check_file_path(file_path)

        # Save task data to ``.txt`` file
        with open(file_path, mode) as f:
            write_msg = f"Writing data to {file_path}"
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                # Check data and get variables
                task._check_sol_not_none()
                vars_num = task.vars_num
                clauses = task.clauses
                sol = task.sol.astype(np.int32)

                # Write data to ``.txt`` file
                f.write(f"{vars_num} clauses ")
                for clause in clauses:
                    for literal in clause:
                        f.write(str(literal) + str(" "))
                    f.write(str("0") + str(" "))
                f.write(str("label") + str(" "))
                f.write(str(" ").join(str(_sol) for _sol in sol))
                f.write("\n")
            f.close()