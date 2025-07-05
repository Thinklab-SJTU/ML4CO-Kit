r"""
REOPT for solving SPCTSPs.

The algorithm plans a tour, executes part of it and then re-optimizes using the C++ ILS algorithm.
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

import subprocess
import os
import numpy as np
from typing import Union, Tuple, List
from scipy.spatial.distance import cdist
from ml4co_kit.solver.spctsp.base import SPCTSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class SPCTSPReoptSolver(SPCTSPSolver):
    r"""
    Solve SPCTSPs using REOPT.

    :param scale, int, the scale factor for coordinates.
    :param time_limit: int, the limit of running time.
    :param cpp_solver_path: str, the path to the C++ solver executable.
    :param runs_per_instance: int, the number of runs for the ILS.
    :param append: str, the method to append the tour.
    """
    def __init__(
        self, 
        scale: int = 1e7, 
        time_limit: int = 1,
        cpp_solver_path: str = "./ml4co_kit/solver/spctsp/reopt_solver",
        runs_per_instance: int = 1,
        append: str = "half"
    ):
        super(SPCTSPReoptSolver, self).__init__(
            solver_type=SOLVER_TYPE.REOPT, scale=scale
        )
        self.time_limit = time_limit
        self.cpp_solver_path = cpp_solver_path
        self.runs_per_instance = runs_per_instance
        self.append = append
    
    def solve(
        self,
        depots: Union[np.ndarray, List[np.ndarray]] = None,
        points: Union[np.ndarray, List[np.ndarray]] = None,
        penalties: Union[np.ndarray, List[np.ndarray]] = None,
        deterministic_prizes: Union[np.ndarray, List[np.ndarray]] = None,
        stochastic_prizes: Union[np.ndarray, List[np.ndarray]] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (costs, routes)
        r"""
        Solve SPCTSP instances using REOPT.
        """        
        # preparation
        self.from_data(
            depots=depots, 
            points=points, 
            penalties=penalties, 
            deterministic_prizes=deterministic_prizes, 
            stochastic_prizes=stochastic_prizes, 
            norm=norm, 
            normalize=normalize
        )
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        costs = list()
        tours = list()
        num_instances = self.points.shape[0]

        # We will use this to convert all floats to integers before writing to file.
        SCALE_FACTOR = 100000.0
        
        for idx in iterative_execution(range, num_instances, self.solve_msg, show_time):
            # Per-instance data
            depot = self.depots[idx]
            customer_points = self.points[idx]
            customer_penalties = self.penalties[idx]
            customer_det_prizes = self.deterministic_prizes[idx]
            customer_stoch_prizes = self.stochastic_prizes[idx]
            
            # Combine depot and customer coordinates for distance matrix calculation
            all_coords = np.vstack([depot, customer_points])
            num_customers = len(customer_points)
            
            # Pre-calculate the full, original distance matrix
            original_dist_matrix = cdist(all_coords, all_coords, 'euclidean')

            # Variables for the re-optimization loop
            final_tour = []
            
            # The re-optimization loop continues as long as not all customers have been visited
            while len(final_tour) < num_customers:
                temp_input_file = f"temp_instance_{idx}.txt"
                try:
                    # State Update
                    visited_mask = np.zeros(len(all_coords), dtype=bool)
                    if final_tour:
                        visited_mask[final_tour] = True
                    
                    current_dist_matrix = original_dist_matrix.copy()
                    if final_tour:
                        last_node_idx = final_tour[-1]
                        current_dist_matrix[0, :] = original_dist_matrix[last_node_idx, :]

                    # Prepare Subproblem
                    unvisited_mask = ~visited_mask
                    sub_dist_matrix = current_dist_matrix[np.ix_(unvisited_mask, unvisited_mask)]
                    sub_penalties = customer_penalties[unvisited_mask[1:]]
                    sub_det_prizes = customer_det_prizes[unvisited_mask[1:]]
                    total_collected_stoch_prize = np.sum(customer_stoch_prizes[np.array(final_tour) - 1]) if final_tour else 0.0
                    prize_to_collect = 1.0 - total_collected_stoch_prize
                    min_prize_scaled = int(max(0, prize_to_collect) * SCALE_FACTOR)
                    max_possible_prize_scaled = int(np.sum(sub_det_prizes) * SCALE_FACTOR)
                    min_prize_scaled = min(min_prize_scaled, max_possible_prize_scaled)

                    # Write Subproblem to File
                    with open(temp_input_file, 'w') as f:
                        # Prizes (for subproblem, depot prize is 0)
                        prizes_full = np.round(np.insert(sub_det_prizes, 0, 0) * SCALE_FACTOR).astype(int)
                        f.write(' '.join(map(str, prizes_full)) + '\n')
                        # Penalties (for subproblem, depot penalty is 0)
                        penalties_full = np.round(np.insert(sub_penalties, 0, 0) * SCALE_FACTOR).astype(int)
                        f.write(' '.join(map(str, penalties_full)) + '\n')
                        # Distance matrix (for subproblem)
                        dist_matrix_int = np.round(sub_dist_matrix * SCALE_FACTOR).astype(int)
                        for row in dist_matrix_int:
                            f.write(' '.join(map(str, row)) + '\n')
                            
                    # Call C++ Solver
                    command = [
                        self.cpp_solver_path,
                        temp_input_file,
                        str(min_prize_scaled),
                        str(self.runs_per_instance)
                    ]
                    
                    result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
                    output = result.stdout

                    # Parse Output and Update Tour
                    sub_tour = []
                    for line in output.strip().split('\n'):
                        if line.startswith("Best Result Cost:"):
                            cost_from_cpp = float(line.split(':')[1].strip())
                        elif line.startswith("Best Result Route:"):
                            full_route = [int(node) for node in line.split(':')[1].strip().split()]
                            if len(full_route) > 2 and full_route[0] == 0 and full_route[-1] == 0:
                                sub_tour = full_route[1:-1]
                    
                    if cost_from_cpp is None:
                        raise RuntimeError("Failed to parse cost from C++ solver output.")

                    if not sub_tour:
                        break
                        
                    unvisited_indices = np.where(unvisited_mask)[0]
                    original_node_indices = unvisited_indices[sub_tour].tolist()
                    
                    # Append part of the new tour to the final tour based on the 'append' strategy
                    if self.append == 'first':
                        final_tour.append(original_node_indices[0])
                    elif self.append == 'half':
                        nodes_to_add = max(1, len(original_node_indices) // 2)
                        final_tour.extend(original_node_indices[:nodes_to_add])
                    else: # 'all'
                        final_tour.extend(original_node_indices)
                        
                finally:
                    # Cleanup 
                    if os.path.exists(temp_input_file):
                        os.remove(temp_input_file)

            final_cost = self.calc_pctsp_cost(depot, customer_points, customer_penalties, customer_stoch_prizes, final_tour)
            
            costs.append(final_cost)
            tours.append(final_tour)

        # format
        costs = np.array(costs)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return costs, self.tours
        

    def __str__(self) -> str:
        return "SPCTSPReoptSolver"