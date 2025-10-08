r"""
OR-Tools Solver for TSP
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
from ml4co_kit.task.routing.tsp import TSPTask


def tsp_ortools(
    task_data: TSPTask,
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