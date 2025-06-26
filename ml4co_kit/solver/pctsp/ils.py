r"""
ILS (C++) for solving PCTSPs.

Solves PCTSPs using Iterated Local Search (ILS).
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
from ml4co_kit.solver.pctsp.base import PCTSPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class PCTSPILSSolver(PCTSPSolver):
    r"""
    Solve PCTSPs using ILS (C++).

    :param scale, int, the scale factor for coordinates.
    :param time_limit: int, the limit of running time.
    :param cpp_solver_path: str, the path to the C++ solver executable.
    :param runs_per_instance: int, the number of runs for the ILS.
    """
    def __init__(
        self, 
        scale: int = 1e7, 
        time_limit: int = 1,
        cpp_solver_path: str = "./ml4co_kit/solver/pctsp/ils_solver",
        runs_per_instance: int = 1,
    ):
        super(PCTSPILSSolver, self).__init__(
            solver_type=SOLVER_TYPE.ILS, scale=scale
        )
        self.time_limit = time_limit
        self.cpp_solver_path = cpp_solver_path
        self.runs_per_instance = runs_per_instance
    
    def solve(
        self,
        depots: Union[np.ndarray, List[np.ndarray]] = None,
        points: Union[np.ndarray, List[np.ndarray]] = None,
        penalties: Union[np.ndarray, List[np.ndarray]] = None,
        prizes: Union[np.ndarray, List[np.ndarray]] = None,
        norm: str = "EUC_2D",
        normalize: bool = False,
        num_threads: int = 1,
        show_time: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (costs, routes)
        r"""
        Solve PCTSP instances using ILS (C++).
        """        
        # preparation
        self.from_data(depots=depots, points=points, penalties=penalties, deterministic_prizes=prizes, norm=norm, normalize=normalize)
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        costs = list()
        tours = list()
        num_points = self.points.shape[0]

        # We will use this to convert all floats to integers before writing to file.
        SCALE_FACTOR = 100000.0
        
        for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
            temp_input_file = f"temp_instance_{idx}.txt"
            
            try:
                # Prepare the input data for the C++ solver
                all_coords = np.vstack([self.depots[idx], self.points[idx]])
                
                dist_matrix = np.round(cdist(all_coords, all_coords, 'euclidean') * SCALE_FACTOR).astype(int)
                prizes_full = np.round(np.insert(self.deterministic_prizes[idx], 0, 0) * SCALE_FACTOR).astype(int)
                penalties_full = np.round(np.insert(self.penalties[idx], 0, 0) * SCALE_FACTOR).astype(int)
                min_prize_scaled = int(1.0 * SCALE_FACTOR)
                
                # Write the properly scaled integer data to a temporary file
                with open(temp_input_file, 'w') as f:
                    # Prizes
                    f.write(' '.join(map(str, prizes_full)) + '\n')
                    # Penalties
                    f.write(' '.join(map(str, penalties_full)) + '\n')
                    # Distance matrix
                    for row in dist_matrix:
                        f.write(' '.join(map(str, row)) + '\n')
                
                print(f"Wrote instance data to {temp_input_file}")
                
                # Call the C++ solver with the scaled min_prize
                command = [
                    self.cpp_solver_path,
                    temp_input_file,
                    str(min_prize_scaled), # Use the scaled integer value for min_prize
                    str(self.runs_per_instance)
                ]
                
                print(f"Executing command: {' '.join(command)}")
                
                result = subprocess.run(
                    command, capture_output=True, text=True, check=True, encoding='utf-8'
                )
                
                print("C++ solver finished. Parsing output...")
                output = result.stdout
                print("--- C++ Output ---")
                print(output)
                print("--------------------")

                # Process the output of the C++ solver
                final_cost = None
                final_route = None
                for line in output.strip().split('\n'):
                    if line.startswith("Best Result Cost:"):
                        final_cost = float(line.split(':')[1].strip())
                    elif line.startswith("Best Result Route:"):
                        full_route = [int(node) for node in line.split(':')[1].strip().split()]
                        if len(full_route) > 2 and full_route[0] == 0 and full_route[-1] == 0:
                            final_route = full_route[1:-1]
                        else:
                            final_route = []

                if final_cost is None or final_route is None:
                    raise RuntimeError("Failed to parse cost or route from C++ solver output.")
                
                # Recalculate cost in Python using original float data for validation.
                total_cost = self.calc_pctsp_cost(self.depots[idx], self.points[idx], self.penalties[idx], self.deterministic_prizes[idx], final_route)
                
                # The cost from C++ is scaled, so scale it back down before comparing.
                cost_from_ils = final_cost / SCALE_FACTOR
                
                print(f"Total cost: {total_cost}, Cost from ILS: {cost_from_ils}")
                assert abs(total_cost - cost_from_ils) <= 1e-4, "Cost is incorrect" # Relaxed tolerance slightly for float precision
                
                costs.append(cost_from_ils)
                tours.append(final_route)
                
            finally:
                # Clean up the temporary input file
                if os.path.exists(temp_input_file):
                    os.remove(temp_input_file)
                    print(f"Cleaned up {temp_input_file}")

        # format
        costs = np.array(costs)
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return costs, self.tours
        

    def __str__(self) -> str:
        return "PCTSPILSSolver"