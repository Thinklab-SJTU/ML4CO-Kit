r"""
MinVarPO Wrapper.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import pathlib
import numpy as np
from typing import Union, List
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.portfolio.minvarpo import MinVarPOTask
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path


class MinVarPOWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(MinVarPOWrapper, self).__init__(
            task_type=TASK_TYPE.MINVARPO, precision=precision
        )
        self.task_list: List[MinVarPOTask] = list()
        
    def from_txt(
        self, 
        file_path: pathlib.Path,
        ref: bool = False,
        overwrite: bool = True,
        show_time: bool = False
    ):
        """Read task data from ``.txt`` file"""
        # Overwrite task list if overwrite is True
        if overwrite:
            self.task_list: List[MinVarPOTask] = list()
        
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                split_line = line.split(" output ")
                
                if len(split_line) != 2:
                    raise ValueError(f"Invalid format in line {idx + 1}: {line}")
                
                data_part = split_line[0]
                sol_part = split_line[1]
                
                # Parse data: returns, cov_matrix, required_returns
                data_parts = data_part.split(" required_returns ")
                if len(data_parts) != 2:
                    raise ValueError(f"Invalid data format in line {idx + 1}: {data_part}")
                
                matrix_part = data_parts[0]
                required_returns = float(data_parts[1])
                
                # Parse matrix part: returns and covariance matrix
                matrix_parts = matrix_part.split(" cov ")
                if len(matrix_parts) != 2:
                    raise ValueError(f"Invalid matrix format in line {idx + 1}: {matrix_part}")
                
                returns_str = matrix_parts[0]
                cov_str = matrix_parts[1]
                
                # Parse returns
                returns = np.array(
                    [float(x) for x in returns_str.split()],
                    dtype=self.precision
                )
                
                # Parse covariance matrix
                cov_values = [float(x) for x in cov_str.split()]
                n_assets = len(returns)
                cov = np.array(cov_values, dtype=self.precision).reshape(n_assets, n_assets)
                
                # Parse solution
                sol = np.array(
                    [float(x) for x in sol_part.split()],
                    dtype=self.precision
                )
                
                # Create task
                if overwrite:
                    minvar_po_task = MinVarPOTask(precision=self.precision)
                else:
                    minvar_po_task = self.task_list[idx]
                
                minvar_po_task.from_data(
                    returns=returns, 
                    cov=cov, 
                    required_returns=required_returns,
                    sol=sol, 
                    ref=ref
                )
                
                if overwrite:
                    self.task_list.append(minvar_po_task)
    
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
                task._check_returns_not_none()
                task._check_cov_not_none()
                task._check_sol_not_none()
                
                returns = task.returns
                cov = task.cov
                required_returns = task.required_returns
                sol = task.sol
                
                # Write data to ``.txt`` file
                # Format: returns cov required_returns output solution
                f.write(" ".join(str(x) for x in returns))
                f.write(" cov ")
                f.write(" ".join(str(x) for x in cov.flatten()))
                f.write(" required_returns ")
                f.write(str(required_returns))
                f.write(" output ")
                f.write(" ".join(str(x) for x in sol))
                f.write("\n")
            f.close()
