r"""
Open Vehicle Routing Problem (OVRP).

OVRP uses the standard CVRP depot-delimited solution format, but the route
closing edge from the last customer back to the depot is not included in the
objective.
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


import numpy as np
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class OVRPTask(CVRPTask):
    def __init__(
        self,
        distance_type: DISTANCE_TYPE = DISTANCE_TYPE.EUC_2D,
        round_type: ROUND_TYPE = ROUND_TYPE.NO,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(OVRPTask, self).__init__(
            distance_type=distance_type,
            round_type=round_type,
            precision=precision,
            threshold=threshold
        )
        self.task_type = TASK_TYPE.OVRP

    def evaluate(self, sol: np.ndarray, check_constr: bool = True) -> np.floating:
        """Evaluate the total open-route distance of the OVRP solution."""
        # Check Constraints
        if check_constr and not self.check_constraints(sol):
            raise ValueError("Invalid solution!")

        # Evaluate without each route's final customer-to-depot delimiter edge.
        total_distance = 0
        for idx in range(len(sol) - 1):
            if sol[idx] != 0 and sol[idx + 1] == 0:
                continue
            cost = self.dist_eval.cal_distance(
                self.coords[sol[idx]], self.coords[sol[idx + 1]]
            )
            total_distance += np.array(cost).astype(self.precision)
        return total_distance
