r"""
Nearest Neighbor Algorithm for CVRP
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
from ml4co_kit.task.routing.cvrp import CVRPTask
from .c_cvrp_nearest import pybind11_cvrp_nearest_segment_impl

def cvrp_nearest(task_data: CVRPTask):
    # Preparation
    depot = task_data.depots.astype(np.float32)
    points = task_data.points.astype(np.float32)
    norm_demands = task_data.norm_demands.astype(np.float32)

    # Build full CVRP solution by repeatedly generating one NN segment.
    # Each segment starts/ends at depot and serves a subset of customers.
    remaining = np.arange(task_data.nodes_num, dtype=np.int32)
    sol = [0]

    while remaining.size > 0:
        sub_points = points[remaining]
        sub_demands = norm_demands[remaining]

        segment = pybind11_cvrp_nearest_segment_impl(depot, sub_points, sub_demands)
        segment = np.asarray(segment, dtype=np.int32)
        
        # segment format: [0, local_node_id+1, ..., 0]
        if segment.size <= 2:
            raise ValueError(
                "CVRP nearest neighbor failed to select any feasible customer. "
                "Please check that normalized demands are <= 1."
            )

        local_nodes = segment[1:-1] - 1
        global_nodes = remaining[local_nodes] + 1  # +1 because depot is 0

        # Append as one route piece, keeping depot separators.
        sol.extend(global_nodes.tolist())
        sol.append(0)

        # Remove served customers from remaining
        served_mask = np.ones(remaining.shape[0], dtype=bool)
        served_mask[local_nodes] = False
        remaining = remaining[served_mask]

    # Store the tour in the task data
    task_data.from_data(sol=np.array(sol, dtype=np.int32), ref=False)