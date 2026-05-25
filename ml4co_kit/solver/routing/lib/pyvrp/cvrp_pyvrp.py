r"""
Solve CVRP using PyVRP
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
from pyvrp.stop import MaxRuntime
from pyvrp.solve import solve as pyvrp_solve
from pyvrp import ProblemData, Depot, Client, VehicleType
from ml4co_kit.task.routing.vrp.cvrp import CVRPTask


def cvrp_pyvrp(
    task_data: CVRPTask,
    time_limit: float = 1.0,
    scale: int = int(1e5),
    seed: int = 1234,
):
    # Get data and scale
    nodes_num = task_data.nodes_num
    demands = (task_data.demands * scale).round().astype(np.int32)
    capacity = int(np.round(task_data.capacity * scale))
    dists = (task_data._get_dists() * scale).round().astype(np.int32)

    # Depots
    depots_coords = task_data.depots * scale
    depots = [
        Depot(
            x=float(depots_coords[0]), 
            y=float(depots_coords[1])
        )
    ]

    # Clients
    points = task_data.points * scale
    clients = [
        Client(
            x=float(points[idx, 0]),
            y=float(points[idx, 1]),
            delivery=[int(demands[idx])],
        ) for idx in range(nodes_num)
    ]

    # Vehicle types
    vehicle_types = [
        VehicleType(
            num_available=nodes_num, # Consider the worst case
            capacity=[capacity],
            start_depot=0,
            end_depot=0
        )
    ]

    # Open routes
    if task_data.cvrp_open:
        dists[:, 0] = 0 # Any client going back to depot is 0

    # Create problem data
    problem_data = ProblemData(
        clients=clients,    
        depots=depots,
        vehicle_types=vehicle_types,
        distance_matrices=[dists],
        duration_matrices=[dists]
    )

    # Stop condition
    stop = MaxRuntime(time_limit)

    # Solve the problem
    result = pyvrp_solve(data=problem_data, stop=stop, seed=seed)
    solution = result.best
    
    # Get solution
    sol = list()
    for route in solution.routes():
        sol.append(0)
        sol.extend(route.visits())
    sol.append(0)
    sol = np.array(sol, dtype=np.int32)
    
    # Store the solution
    task_data.from_data(sol=sol, ref=False)
