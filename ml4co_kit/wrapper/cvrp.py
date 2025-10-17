r"""
CVRP Wrapper.
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
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class CVRPWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(CVRPWrapper, self).__init__(
            task_type=TASK_TYPE.CVRP, precision=precision
        )
        self.task_list: List[CVRPTask] = list()
        
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
            self.task_list: List[CVRPTask] = list()
        
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                split_line_0 = line.split("depots ")[1]
                split_line_1 = split_line_0.split(" points ")
                depot = split_line_1[0]
                split_line_2 = split_line_1[1].split(" demands ")
                points = split_line_2[0]
                split_line_3 = split_line_2[1].split(" capacity ")
                demands = split_line_3[0]
                split_line_4 = split_line_3[1].split(" output ")
                capacity = split_line_4[0]
                tour = split_line_4[1]
                
                # Parse depot coordinates
                depot = depot.split(" ")
                depot = np.array([float(depot[0]), float(depot[1])], dtype=self.precision)
                
                # Parse points coordinates
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ], dtype=self.precision
                )
                
                # Parse demands
                demands = demands.split(" ")
                demands = np.array(
                    [float(demands[i]) for i in range(len(demands))], dtype=self.precision
                )
                
                # Parse capacity
                capacity = float(capacity)
                
                # Parse tour
                tour = tour.split(" ")
                tour = np.array([int(t) for t in tour])
                
                # Create a new task and add it to ``self.task_list``
                if overwrite:
                    cvrp_task = CVRPTask(
                        distance_type=distance_type,
                        round_type=round_type,
                        precision=self.precision
                    )
                else:
                    cvrp_task = self.task_list[idx]
                cvrp_task.from_data(
                    depots=depot, points=points, demands=demands, capacity=capacity,
                    sol=tour, ref=ref, normalize=normalize
                )
                if overwrite:
                    self.task_list.append(cvrp_task)
    
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
                task._check_depots_not_none()
                task._check_points_not_none()
                task._check_demands_not_none()
                task._check_capacity_not_none()
                task._check_sol_not_none()
                
                depot = task.depots
                points = task.points
                demands = task.demands
                capacity = task.capacity
                sol = task.sol

                # Write data to ``.txt`` file
                f.write("depots " + str(depot[0]) + str(" ") + str(depot[1]))
                f.write(" points" + str(" "))
                f.write(
                    " ".join(
                        str(x) + str(" ") + str(y)
                        for x, y in points
                    )
                )
                f.write(" demands " + str(" ").join(str(demand) for demand in demands))
                f.write(" capacity " + str(capacity))
                f.write(str(" output "))
                f.write(str(" ").join(str(node_idx) for node_idx in sol))
                f.write("\n")
            f.close()
    
    def from_vrplib_folder(
        self, 
        vrp_folder_path: pathlib.Path = None,
        sol_folder_path: pathlib.Path = None,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        ref: bool = False,
        overwrite: bool = True,
        normalize: bool = False,
        show_time: bool = False                  
    ):
        """Read task data from folder (to support VRPLIB)"""
        # Overwrite task list if overwrite is True
        if overwrite:
            self.task_list: List[CVRPTask] = list()
        
        # Check inconsistent file names between vrp and sol files
        if vrp_folder_path is not None and sol_folder_path is not None:
            vrp_files = os.listdir(vrp_folder_path)
            vrp_files.sort()
            sol_files = os.listdir(sol_folder_path)
            sol_files.sort()
            vrp_name_list = [file.split(".")[0] for file in vrp_files]
            sol_name_list = [file.split(".")[0] for file in sol_files]
            if vrp_name_list != sol_name_list:
                raise ValueError("Inconsistent file names between vrp and sol files.")
            
        # Get file paths and number of instances
        num_instance = None
        if vrp_folder_path is not None:
            vrp_files = os.listdir(vrp_folder_path)
            vrp_files.sort()
            vrp_files_path = [
                os.path.join(vrp_folder_path, file) 
                for file in vrp_files if file.endswith(".vrp")
            ]
            num_instance = len(vrp_files_path)
        if sol_folder_path is not None:
            sol_files = os.listdir(sol_folder_path)
            sol_files.sort()
            sol_files_path = [
                os.path.join(sol_folder_path, file) 
                for file in sol_files if file.endswith(".sol")
            ]
            num_instance = len(sol_files_path)
        
        # Set None to file paths if not provided
        if num_instance is None:
            raise ValueError(
                "``vrp_folder_path`` and ``sol_folder_path`` cannot be None at the same time."
            )
        elif num_instance == 0:
            raise ValueError("No instance found in the folder.")
        else:
            if vrp_folder_path is None:
                vrp_files_path = [None] * num_instance
            if sol_folder_path is None:
                sol_files_path = [None] * num_instance
        
        # Read task data from VRPLIB files
        if vrp_folder_path is None:
            load_msg = f"Loading solution from {sol_folder_path}"
        else:
            if sol_folder_path is None:
                load_msg = f"Loading data from {vrp_folder_path}"
            else:
                load_msg = (
                    f"Loading data from {vrp_folder_path} and "
                    f"solution from {sol_folder_path}"
                )
        
        for idx, (vrp_file_path, sol_file_path) in tqdm_by_time(
            enumerate(zip(vrp_files_path, sol_files_path)), load_msg, show_time
        ):
            if overwrite:
                cvrp_task = CVRPTask(round_type=round_type, precision=self.precision)
            else:
                cvrp_task = self.task_list[idx]
            cvrp_task.from_vrplib(
                vrp_file_path=vrp_file_path, sol_file_path=sol_file_path, 
                ref=ref, normalize=normalize
            )
            if overwrite:
                self.task_list.append(cvrp_task)
        
    def to_vrplib_folder(
        self, 
        vrp_folder_path: pathlib.Path = None, 
        sol_folder_path: pathlib.Path = None, 
        show_time: bool = False,
        sequential_orderd: bool = True
    ):
        # Write problem of task data (.vrp)
        if vrp_folder_path is not None:
            os.makedirs(vrp_folder_path, exist_ok=True)
            write_msg = f"Writing data to {vrp_folder_path} and {sol_folder_path}"
            idx = 1  # Initialize idx
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                if sequential_orderd:
                    idx_str = f"{idx:08d}"
                    vrp_file_path = os.path.join(vrp_folder_path, f"{idx_str}.vrp")
                    idx += 1  # Increment idx for the next task
                else:
                    vrp_file_path = os.path.join(vrp_folder_path, f"{task.name}.vrp")
                task.to_vrplib(vrp_file_path=vrp_file_path)
        
        # Write solution of task data (.sol)
        if sol_folder_path is not None:
            os.makedirs(sol_folder_path, exist_ok=True)
            write_msg = f"Writing solution to {sol_folder_path}"
            idx = 1  # Initialize idx
            for task in tqdm_by_time(self.task_list, write_msg, show_time):
                if sequential_orderd:
                    idx_str = f"{idx:08d}"
                    sol_file_path = os.path.join(sol_folder_path, f"{idx_str}.sol")
                    idx += 1  # Increment idx for the next task
                else:
                    sol_file_path = os.path.join(sol_folder_path, f"{task.name}.sol")
                task.to_vrplib(sol_file_path=sol_file_path)