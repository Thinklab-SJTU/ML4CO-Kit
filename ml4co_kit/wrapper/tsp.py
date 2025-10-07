r"""
TSP Wrapper.
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
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class TSPWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(TSPWrapper, self).__init__(
            task_type=TASK_TYPE.TSP, precision=precision
        )
        self.task_list: List[TSPTask] = list()
        
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
            self.task_list: List[TSPTask] = list()
        
        # Read task data from ``.txt`` file
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                split_line = line.split(" output ")
                points = split_line[0]
                tour = split_line[1]
                tour = tour.split(" ")
                tour = np.array([int(t) for t in tour])
                tour -= 1
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ], dtype=self.precision
                )
                
                # Create a new task and add it to ``self.task_list``
                if overwrite:
                    tsp_task = TSPTask(
                        distance_type=distance_type,
                        round_type=round_type,
                        precision=self.precision
                    )
                else:
                    tsp_task = self.task_list[idx]
                tsp_task.from_data(
                    points=points, sol=tour, ref=ref, normalize=normalize
                )
                if overwrite:
                    self.task_list.append(tsp_task)
    
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
                task._check_points_not_none()
                task._check_sol_not_none()
                points = task.points
                sol = task.sol

                # Write data to ``.txt`` file
                f.write(" ".join(str(x) + str(" ") + str(y) for x, y in points))
                f.write(str(" ") + str("output") + str(" "))
                f.write(str(" ").join(str(node_idx + 1) for node_idx in sol))
                f.write("\n")
            f.close()
    
    def from_tsplib_folder(
        self, 
        tsp_folder_path: pathlib.Path = None,
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
            self.task_list: List[TSPTask] = list()
        
        # Check inconsistent file names between tsp and tour files
        if tsp_folder_path is not None and tour_folder_path is not None:
            tsp_files = os.listdir(tsp_folder_path)
            tsp_files.sort()
            tour_files = os.listdir(tour_folder_path)
            tour_files.sort()
            tsp_name_list = [file.split(".")[0] for file in tsp_files]
            tour_name_list = [file.split(".")[0] for file in tour_files]
            if tsp_name_list != tour_name_list:
                raise ValueError("Inconsistent file names between tsp and tour files.")
            
        # Get file paths and number of instances
        num_instance = None
        if tsp_folder_path is not None:
            tsp_files = os.listdir(tsp_folder_path)
            tsp_files.sort()
            tsp_files_path = [
                os.path.join(tsp_folder_path, file) 
                for file in tsp_files if file.endswith(".tsp")
            ]
            num_instance = len(tsp_files_path)
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
                "``tsp_folder_path`` and ``tour_folder_path`` cannot be None at the same time."
            )
        elif num_instance == 0:
            raise ValueError("No instance found in the folder.")
        else:
            if tsp_folder_path is None:
                tsp_files_path = [None] * num_instance
            if tour_folder_path is None:
                tour_files_path = [None] * num_instance
        
        # Read task data from TSPLIB files
        if tsp_folder_path is None:
            load_msg = f"Loading solution from {tour_folder_path}"
        else:
            if tour_folder_path is None:
                load_msg = f"Loading data from {tsp_folder_path}"
            else:
                load_msg = (
                    f"Loading data from {tsp_folder_path} and "
                    f"solution from {tour_folder_path}"
                )
        
        for idx, (tsp_file_path, tour_file_path) in tqdm_by_time(
            enumerate(zip(tsp_files_path, tour_files_path)), load_msg, show_time
        ):
            if overwrite:
                tsp_task = TSPTask(round_type=round_type, precision=self.precision)
            else:
                tsp_task = self.task_list[idx]
            tsp_task.from_tsplib(
                tsp_file_path=tsp_file_path, tour_file_path=tour_file_path, 
                ref=ref, normalize=normalize
            )
            if overwrite:
                self.task_list.append(tsp_task)
        
    def to_tsplib_folder(
        self, 
        tsp_folder_path: pathlib.Path = None, 
        tour_folder_path: pathlib.Path = None, 
        show_time: bool = False,
        sequential_orderd: bool = True
    ):
        # Write problem of task data (.tsp)
        if tsp_folder_path is not None:
            os.makedirs(tsp_folder_path, exist_ok=True)
            write_msg = f"Writing data to {tsp_folder_path} and {tour_folder_path}"
            idx = 1  # Initialize idx
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                if sequential_orderd:
                    idx_str = f"{idx:08d}"
                    tsp_file_path = os.path.join(tsp_folder_path, f"{idx_str}.tsp")
                    idx += 1  # Increment idx for the next task
                else:
                    tsp_file_path = os.path.join(tsp_folder_path, f"{task.name}.tsp")
                task.to_tsplib(tsp_file_path=tsp_file_path)
        
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