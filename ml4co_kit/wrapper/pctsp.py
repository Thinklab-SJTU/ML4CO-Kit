r"""
PCTSP Wrapper.
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
from ml4co_kit.task.routing.pctsp import PCTSPTask
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.utils.file_utils import check_file_path
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class PCTSPWrapper(WrapperBase):
    def __init__(
        self, precision: Union[np.float32, np.float64] = np.float32
    ):
        super(PCTSPWrapper, self).__init__(
            task_type=TASK_TYPE.PCTSP, precision=precision
        )
        self.task_list: List[PCTSPTask] = list()
        
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
            self.task_list: List[PCTSPTask] = list()
        
        # Read task data from ``.txt`` file
        with open(file_path, "r") as file:
            load_msg = f"Loading data from {file_path}"
            for idx, line in tqdm_by_time(enumerate(file), load_msg, show_time):
                # Load data
                line = line.strip()
                split_line_0 = line.split("depots ")[1]
                split_line_1 = split_line_0.split(" points ")
                depots = split_line_1[0]
                split_line_2 = split_line_1[1].split(" penalties ")
                points = split_line_2[0]
                split_line_3 = split_line_2[1].split(" prizes ")
                penalties = split_line_3[0]
                split_line_4 = split_line_3[1].split(" required_prize ")
                prizes = split_line_4[0]
                split_line_5 = split_line_4[1].split(" output ")
                required_prize = split_line_5[0]
                tour = split_line_5[1]
                
                # Parse depot coordinates
                depots = depots.split(" ")
                depots = np.array([float(depots[0]), float(depots[1])], dtype=self.precision)
                
                # Parse points coordinates
                points = points.split(" ")
                points = np.array(
                    [
                        [float(points[i]), float(points[i + 1])]
                        for i in range(0, len(points), 2)
                    ], dtype=self.precision
                )
                
                # Parse penalties
                penalties = penalties.split(" ")
                penalties = np.array([
                    float(penalties[i]) for i in range(len(penalties))
                ], dtype=self.precision)
                
                # Parse prizes
                prizes = prizes.split(" ")
                prizes = np.array(
                    [float(prizes[i]) for i in range(len(prizes))], dtype=self.precision
                )

                # Parse required_prize
                required_prize = float(required_prize)
                
                # Parse tour
                tour = tour.split(" ")
                tour = np.array(
                    [int(tour[i]) for i in range(len(tour))]
                )
                tour -= 1
                
                # Create a new task and add it to ``self.task_list``
                if overwrite:
                    pctsp_task = PCTSPTask(
                        distance_type=distance_type,
                        round_type=round_type,
                        precision=self.precision
                    )
                else:
                    pctsp_task = self.task_list[idx]
                pctsp_task.from_data(
                    depots=depots, points=points, penalties=penalties, prizes=prizes, 
                    required_prize=required_prize, sol=tour, ref=ref, normalize=normalize
                )
                if overwrite:
                    self.task_list.append(pctsp_task)
    
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
                task._check_penalties_not_none()
                task._check_prizes_not_none()
                task._check_required_prize_not_none()
                task._check_sol_not_none()
                
                depots = task.depots
                points = task.points
                prizes = task.prizes
                penalties = task.penalties
                required_prize = task.required_prize
                sol = task.sol
                
                # Write data to ``.txt`` file
                f.write("depots " + str(depots[0]) + str(" ") + str(depots[1]))
                f.write(" points" + str(" "))
                f.write(
                    " ".join(
                        str(x) + str(" ") + str(y)
                        for x, y in points
                    )
                )
                f.write(" penalties " + str(" ").join(str(penalty) for penalty in penalties))
                f.write(" prizes " + str(" ").join(str(prize) for prize in prizes))
                f.write(" required_prize " + str(required_prize))
                f.write(str(" output "))
                f.write(str(" ").join(str(node_idx + 1) for node_idx in sol))
                f.write("\n")
            f.close()