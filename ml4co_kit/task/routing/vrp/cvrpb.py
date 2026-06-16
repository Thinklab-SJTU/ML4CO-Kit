r"""
Capacitated Vehicle Routing Problem with Backhauls (CVRPB).

Linehaul customers have ``demands`` > 0 (delivery from depot);
backhaul customers have ``demands`` < 0 (pickup to depot).
On each route, all linehaul visits precede all backhaul visits.
Linehaul load must not exceed capacity; backhaul load must not
exceed the remaining capacity (capacity - linehaul_load).
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


import pathlib
import numpy as np
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.vrp.cvrp import CVRPTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class CVRPBTask(CVRPTask):
    def __init__(
        self,
        cvrp_open: bool = False,
        mixed_backhaul: bool = False,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-4,
    ):
        # Super Initialization
        super(CVRPBTask, self).__init__(
            cvrp_open=cvrp_open,
            distance_type=distance_type,
            round_type=round_type,
            precision=precision,
            threshold=threshold,
        )

        # Set Task Type and Mixed Backhaul
        self.task_type = TASK_TYPE.CVRPB
        self.mixed_backhaul = mixed_backhaul

    def check_constraints(self, sol: np.ndarray) -> bool:
        """Check if the solution is valid for backhauls."""
        # Every tour starts and ends with the depot
        if sol[0] != 0 or sol[-1] != 0:
            return False
        
        # Every node is visited exactly once
        ordered_sol = np.sort(sol)[-self.nodes_num:]
        if not np.all(ordered_sol == (np.arange(self.nodes_num) + 1)):
            return False

        # Split Tours
        demands = self.demands
        capacity = self.capacity
        split_tours = self._split_tours(sol)

        # For each split tour, check:
        for split_idx in range(len(split_tours)):
            # Get the split tour
            split_tour: np.ndarray = split_tours[split_idx][1:]

            # Check the constraint B or MB
            if self.mixed_backhaul:
                if not self._check_route_mb(
                    route=split_tour,
                    demands=demands,
                    capacity=capacity,
                    threshold=self.threshold
                ):
                    return False
            else:
                if not self._check_route_b(
                    route=split_tour,
                    demands=demands,
                    capacity=capacity,
                    threshold=self.threshold
                ):
                    return False

        # If all constraints are satisfied, return True
        return True

    def render(
        self, 
        save_path: pathlib.Path, 
        with_sol: bool = True,
        figsize: tuple = (5, 5),
        node_color: str = "darkblue",
        edge_color: str = "darkblue",
        node_size: int = 50,
    ):
        pass
        # TODO: Implement the render method for CVRPB