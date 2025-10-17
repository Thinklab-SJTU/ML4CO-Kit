r"""
OR-Tools Solver for PCTSP
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
from ortools.constraint_solver import pywrapcp
from ml4co_kit.task.routing.pctsp import PCTSPTask


def pctsp_ortools(
    task_data: PCTSPTask,
    ortools_scale: int = 1e6,
    search_parameters: pywrapcp.DefaultRoutingSearchParameters = None
):
    # Preparation
    data = {}
    dists = task_data._get_dists()
    dists = (dists * ortools_scale).astype(np.int32)
    data["distance_matrix"] = dists
    data["num_vehicles"] = 1
    data["depot"] = 0
    penalties = np.round(task_data.penalties * ortools_scale).astype(np.int32)
    norm_prizes = np.round(task_data.norm_prizes * ortools_scale).astype(np.int32)
    
    # Create manager
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create routing model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add Penalties for Unvisited Nodes (Disjunctions)
    # Note: Nodes for disjunctions are 1-indexed (excluding depot) relative to their original problem indices
    # In OR-Tools, the node index `c + 1` corresponds to the actual location index.
    # Penalties are applied to nodes from 1 to num_locations - 1
    for i, p_val in enumerate(penalties):
        # routing.AddDisjunction takes a list of node indices (managed by manager)
        # The actual node index for the i-th penalty (0-indexed penalty array) is (i + 1)
        # manager.NodeToIndex converts the problem node index to the internal routing index
        routing.AddDisjunction([manager.NodeToIndex(i + 1)], int(p_val))

    # Add Prize Constraint
    # Method: Use a dimension to record prize, and check in the solution filter
    def prize_callback(index):
        node = manager.IndexToNode(index)
        return 0 if node == 0 else norm_prizes[node-1]  # Scale up to avoid precision problem
    prize_callback_index = routing.RegisterUnaryTransitCallback(prize_callback)
    
    # Add Prize Dimension
    routing.AddDimension(
        prize_callback_index,
        0,  # null capacity slack (no prize accumulated when not moving)
        int(sum(norm_prizes)), # total capacity, i.e., max possible prize
        True,  # start cumul to zero
        "Prize"
    )
    prize_dimension = routing.GetDimensionOrDie("Prize")
    
    # The solution must satisfy prize >= ortools_scale (i.e., >=1)
    routing.solver().Add(
        prize_dimension.CumulVar(routing.End(0)) >= int(ortools_scale)
    )
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Solution -> tour
    tour = list()
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        index = solution.Value(routing.NextVar(index))
        tour.append(node)
    tour.append(0)
    
    # Store the tour in the task_data
    task_data.from_data(sol=np.array(tour), ref=False)