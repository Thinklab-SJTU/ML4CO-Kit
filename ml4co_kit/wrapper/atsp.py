r"""
ATSP Wrapper.
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
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.task.routing.atsp import ATSPTask
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class ATSPWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(ATSPWrapper, self).__init__(
            task_type=TASK_TYPE.ATSP, precision=precision
        )
        self.task_list: List[ATSPTask] = list()
        
    def from_txt(
        self, 
        file_path: pathlib.Path,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        ref: bool = False,
        overwrite: bool = True,
        normalize: bool = False,
        show_time: bool = False
    ):
        """Read task data from ``.txt`` file"""
        # Overwrite task list if overwrite is True
        if overwrite:
            self.task_list: List[ATSPTask] = list()
        
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                split_line = line.split(" output ")
                dists = split_line[0]
                tour = split_line[1]
                tour = tour.split(" ")
                tour = np.array([int(t) for t in tour])
                tour -= 1
                dists = dists.split(" ")
                dists.append('')           
                dists = np.array(
                    [float(dists[2*i]) for i in range(len(dists) // 2)], 
                    dtype=self.precision
                )
                num_nodes = int(np.sqrt(len(dists)))
                dists = dists.reshape(num_nodes, num_nodes)
                
                # Create a new task and add it to ``self.task_list``
                if overwrite:
                    atsp_task = ATSPTask(
                        distance_type=distance_type,
                        round_type=round_type,
                        precision=self.precision
                    )
                else:
                    atsp_task = self.task_list[idx]
                atsp_task.from_data(
                    dists=dists, sol=tour, ref=ref, normalize=normalize
                )
                if overwrite:
                    self.task_list.append(atsp_task)
    
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
                task._check_dists_not_none()
                task._check_sol_not_none()
                dists = task.dists
                sol = task.sol

                # Write data to ``.txt`` file
                f.write(" ".join(str(x) + str(" ") for x in dists.reshape(-1)))
                f.write(str(" ") + str("output") + str(" "))
                f.write(str(" ").join(str(node_idx + 1) for node_idx in sol))
                f.write("\n")
            f.close()
    
    def from_tsplib_folder(
        self, 
        atsp_folder_path: pathlib.Path = None,
        tour_folder_path: pathlib.Path = None,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        ref: bool = False,
        overwrite: bool = True,
        normalize: bool = False,
        show_time: bool = False                  
    ):
        """Read task data from folder (to support TSPLIB)"""
        # Overwrite task list if overwrite is True
        if overwrite:
            self.task_list: List[ATSPTask] = list()
        
        # Check inconsistent file names between atsp and tour files
        if atsp_folder_path is not None and tour_folder_path is not None:
            atsp_files = os.listdir(atsp_folder_path)
            atsp_files.sort()
            tour_files = os.listdir(tour_folder_path)
            tour_files.sort()
            atsp_name_list = [file.split(".")[0] for file in atsp_files]
            tour_name_list = [file.split(".")[0] for file in tour_files]
            if atsp_name_list != tour_name_list:
                raise ValueError("Inconsistent file names between atsp and tour files.")
            
        # Get file paths and number of instances
        num_instance = None
        if atsp_folder_path is not None:
            atsp_files = os.listdir(atsp_folder_path)
            atsp_files.sort()
            atsp_files_path = [
                os.path.join(atsp_folder_path, file) 
                for file in atsp_files if file.endswith(".atsp") or file.endswith(".tsp")
            ]
            num_instance = len(atsp_files_path)
        if tour_folder_path is not None:
            tour_files = os.listdir(tour_folder_path)
            tour_files.sort()
            tour_files_path = [
                os.path.join(tour_folder_path, file) 
                for file in tour_files if file.endswith(".tour")
            ]
            num_instance = len(tour_files_path)
        
        # Set None to file paths if not provided
        if num_instance is None:
            raise ValueError(
                "``atsp_folder_path`` and ``tour_folder_path`` cannot be None at the same time."
            )
        elif num_instance == 0:
            raise ValueError("No instance found in the folder.")
        else:
            if atsp_folder_path is None:
                atsp_files_path = [None] * num_instance
            if tour_folder_path is None:
                tour_files_path = [None] * num_instance
        
        # Read task data from TSPLIB files
        if atsp_folder_path is None:
            load_msg = f"Loading solution from {tour_folder_path}"
        else:
            if tour_folder_path is None:
                load_msg = f"Loading data from {atsp_folder_path}"
            else:
                load_msg = (
                    f"Loading data from {atsp_folder_path} and "
                    f"solution from {tour_folder_path}"
                )
        
        for idx, (atsp_file_path, tour_file_path) in tqdm_by_time(
            enumerate(zip(atsp_files_path, tour_files_path)), load_msg, show_time
        ):
            if overwrite:
                atsp_task = ATSPTask(round_type=round_type, precision=self.precision)
            else:
                atsp_task = self.task_list[idx]
            atsp_task.from_tsplib(
                atsp_file_path=atsp_file_path, tour_file_path=tour_file_path, 
                ref=ref, normalize=normalize
            )
            if overwrite:
                self.task_list.append(atsp_task)
        
    def to_tsplib_folder(
        self, 
        atsp_folder_path: pathlib.Path = None, 
        tour_folder_path: pathlib.Path = None, 
        show_time: bool = False,
        sequential_orderd: bool = True,
    ):
        # Write problem of task data (.atsp)
        if atsp_folder_path is not None:
            os.makedirs(atsp_folder_path, exist_ok=True)
            write_msg = f"Writing data to {atsp_folder_path} and {tour_folder_path}"
            idx = 1  # Initialize idx
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                if sequential_orderd:
                    idx_str = f"{idx:08d}"
                    atsp_file_path = os.path.join(atsp_folder_path, f"{idx_str}.atsp")
                    idx += 1  # Increment idx for the next task
                else:
                    atsp_file_path = os.path.join(atsp_folder_path, f"{task.name}.atsp")
                task.to_tsplib(atsp_file_path=atsp_file_path)
        
        # Write solution of task data (.tour)
        if tour_folder_path is not None:
            os.makedirs(tour_folder_path, exist_ok=True)
            write_msg = f"Writing solution to {tour_folder_path}"
            idx = 1  # Initialize idx
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                if sequential_orderd:
                    idx_str = f"{idx:08d}"
                    tour_file_path = os.path.join(tour_folder_path, f"{idx_str}.tour")
                    idx += 1  # Increment idx for the next task
                else:
                    tour_file_path = os.path.join(tour_folder_path, f"{task.name}.tour")
                task.to_tsplib(tour_file_path=tour_file_path)