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

        # Set Task Type
        self.task_type = TASK_TYPE.CVRPB

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
        # 1. if the linehaul demand is served before the backhaul demand
        # 2. if the linehaul demand is within the capacity
        # 3. if the backhaul demand is within the capacity
        for split_idx in range(len(split_tours)):
            # Get the split tour and the demands on the split tour
            split_tour: np.ndarray = split_tours[split_idx][1:]
            route_demands = demands[split_tour.astype(int) - 1]

            # 1. Linehaul before backhaul
            is_linehaul = route_demands >= 0
            is_backhaul = route_demands < 0
            if np.any(is_backhaul):
                first_backhaul_idx = np.flatnonzero(is_backhaul)[0]
                if np.any(is_linehaul[first_backhaul_idx + 1:]):
                    return False

            # 2. Linehaul load <= vehicle capacity
            linehaul_load = np.sum(route_demands[is_linehaul], dtype=self.precision)
            if linehaul_load > capacity + self.threshold:
                return False

            # 3. Backhaul load <= vehicle capacity
            backhaul_load = np.sum(-route_demands[is_backhaul], dtype=self.precision)
            if backhaul_load > capacity + self.threshold:
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