r"""
CVRPBL Wrapper.
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
from ml4co_kit.wrapper.base import WrapperBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.vrp.cvrpbl import CVRPBLTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class CVRPBLWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(CVRPBLWrapper, self).__init__(
            task_type=TASK_TYPE.CVRPBL, precision=precision
        )
        self.task_list: List[CVRPBLTask] = list()
        self.task_class = CVRPBLTask
        
    def from_txt(
        self, 
        file_path: pathlib.Path,
        cvrp_open: bool = False,
        mixed_backhaul: bool = False,
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
            self.task_list: List[CVRPBLTask] = list()
        
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
                split_line_4 = split_line_3[1].split(" max_route_length ")
                capacity = split_line_4[0]
                split_line_5 = split_line_4[1].split(" output ")
                max_route_length = split_line_5[0]
                tour = split_line_5[1]
                
                # Parse depot coordinates
                depot = depot.split(" ")
                depot = np.array([
                    float(depot[0]), float(depot[1])], 
                    dtype=self.precision
                )
                
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
                    [float(demands[i]) for i in range(len(demands))], 
                    dtype=self.precision
                )
                
                # Parse max route length
                max_route_length = float(max_route_length)
                
                # Parse capacity
                capacity = float(capacity)
                
                # Parse tour
                tour = tour.split(" ")
                tour = np.array([int(t) for t in tour])
                
                # Create a new task and add it to ``self.task_list``
                if overwrite:
                    cvrpbl_task = CVRPBLTask(
                        cvrp_open=cvrp_open,
                        mixed_backhaul=mixed_backhaul,
                        distance_type=distance_type,
                        round_type=round_type,
                        precision=self.precision
                    )
                else:
                    cvrpbl_task = self.task_list[idx]
                cvrpbl_task.from_data(
                    depots=depot, points=points, 
                    demands=demands, capacity=capacity, 
                    max_route_length=max_route_length,
                    sol=tour, ref=ref, normalize=normalize
                )
                if overwrite:
                    self.task_list.append(cvrpbl_task)
    
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
                task._check_max_route_length_not_none()
                task._check_sol_not_none()
                
                depot = task.depots
                points = task.points
                demands = task.demands
                capacity = float(task.capacity)
                max_route_length = task.max_route_length
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
                f.write(str(" max_route_length ") + str(max_route_length))
                f.write(str(" output "))
                f.write(str(" ").join(str(node_idx) for node_idx in sol))
                f.write("\n")
            f.close()