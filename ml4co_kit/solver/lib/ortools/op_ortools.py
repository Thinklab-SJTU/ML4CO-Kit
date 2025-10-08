r"""
OR-Tools Solver for OP
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
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ml4co_kit.task.routing.op import OPTask


def op_ortools(
    task_data: OPTask,
    ortools_scale: int = 1e6,
    search_parameters: pywrapcp.DefaultRoutingSearchParameters = None
):
    # Prepare data
    dists = np.asarray(task_data._get_dists())  # expected shape: (N, N)
    prizes = np.asarray(task_data.prizes)  # could be length N (including depot) or N-1 (excluding depot)
    max_length = float(task_data.max_length)
    N = int(dists.shape[0])

    # Scale to ints (use int64 to be safe)
    scale = int(ortools_scale)
    # convert to python nested lists of ints (OR-Tools prefers Python ints)
    dist_matrix = (dists * scale).round().astype(np.int64).tolist()
    scaled_prizes = (np.round(prizes * scale).astype(np.int64) if prizes.size > 0 else np.array([], dtype=np.int64))

    # Build prizes_per_node of length N (prize at depot set to 0)
    prizes_per_node = np.zeros(N, dtype=np.int64)
    if scaled_prizes.size == N:
        prizes_per_node = scaled_prizes.copy()
    elif scaled_prizes.size == N - 1:
        # Common convention: prizes excludes depot; assume depot is index 0
        prizes_per_node[1:] = scaled_prizes
    elif scaled_prizes.size == 0:
        prizes_per_node[:] = 0
    else:
        # Fallback: try to fill as much as possible (robustness)
        prizes_per_node[1:1 + scaled_prizes.size] = scaled_prizes[:N - 1]

    scaled_max_length = int(round(max_length * scale))

    # OR-Tools setup
    manager = pywrapcp.RoutingIndexManager(N, 1, 0)  # single vehicle, depot = 0
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Length dimension: cumulative distance, enforce max length.
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        scaled_max_length,  # capacity = max tour length
        True,  # start cumul to zero
        "Length",
    )
    length_dimension = routing.GetDimensionOrDie("Length")
    # Ensure route end respects max length (redundant with capacity but explicit).
    length_dimension.CumulVar(routing.End(0)).SetMax(scaled_max_length)

    # Optional nodes: add disjunctions with penalty = prize(node).
    # If prize==0 for some nodes, skipping them has zero penalty (equivalent to always optional)
    for node in range(1, N):  # skip depot (0)
        penalty = int(prizes_per_node[node])
        if penalty < 0:
            penalty = 0
        # AddDisjunction takes routing indices (use manager.NodeToIndex).
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Solution -> tour
    if solution:
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            tour.append(int(node))
            index = solution.Value(routing.NextVar(index))
        # append final node (should be depot 0)
        tour.append(int(manager.IndexToNode(index)))
        task_data.from_data(sol=np.array(tour), ref=False)
    else:
        # fallback: no feasible route found (only depot)
        raise ValueError("no feasible solution has been found")
        
