import os
import uuid
import pathlib
import subprocess
import numpy as np


BASE_PATH = pathlib.Path(__file__).parent
SOLVER_PATH = pathlib.Path(__file__).parent / "spctsp_reopt"
TMP_PATH = pathlib.Path(__file__).parent / "tmp"

# Determining whether the solvers have been built
if not os.path.exists(SOLVER_PATH):
    ori_dir = os.getcwd()
    os.chdir(BASE_PATH)
    os.system("make")
    os.chdir(ori_dir)


# make tmp dir
if not os.path.exists(TMP_PATH):
    os.makedirs(TMP_PATH)


def spctsp_reopt_solve(
    dist_matrix: np.ndarray, 
    penalties: np.ndarray, 
    norm_prizes: np.ndarray, 
    stochastic_norm_prizes: np.ndarray,  
    runs_per_instance: int,
    scale: int,
    append_strategy: str
) -> np.ndarray:
    # tmp file name
    tmp_name = uuid.uuid4().hex[:9]
    tmp_file_path = TMP_PATH / f"{tmp_name}.txt"
    
    # The re-optimization loop continues as long as not all customers have been visited
    tour = list()
    nodes_num = len(dist_matrix)
    remain_prize_to_collect = scale
    while remain_prize_to_collect > 0:
        # State Update
        visited_mask = np.zeros(nodes_num, dtype=bool)
        if tour:
            visited_mask[tour] = True
        
        # Sub-problem        
        current_dist_matrix = dist_matrix.copy()
        if tour:
            last_node_idx = tour[-1]
            current_dist_matrix[0, :] = dist_matrix[last_node_idx, :]
        unvisited_mask = ~visited_mask
        sub_dist_matrix = current_dist_matrix[np.ix_(unvisited_mask, unvisited_mask)]
        sub_penalties = penalties[unvisited_mask[1:]]
        sub_det_prizes = norm_prizes[unvisited_mask[1:]]
        total_collected_stoch_prize = np.sum(stochastic_norm_prizes[np.array(tour) - 1]) if tour else 0.0
        if total_collected_stoch_prize >= 1.0:
            return tour
        remain_prize_to_collect = (1.0 - total_collected_stoch_prize) * scale
        min_prize_scaled = int(max(0, remain_prize_to_collect))
        max_possible_prize_scaled = int(np.sum(sub_det_prizes) * scale)
        min_prize_scaled = min(min_prize_scaled, max_possible_prize_scaled)

        # Write Subproblem to File
        with open(tmp_file_path, 'w') as f:
            # Prizes (for subproblem, depot prize is 0)
            prizes_full = np.round(np.insert(sub_det_prizes, 0, 0) * scale).astype(int)
            f.write(' '.join(map(str, prizes_full)) + '\n')
            # Penalties (for subproblem, depot penalty is 0)
            penalties_full = np.round(np.insert(sub_penalties, 0, 0) * scale).astype(int)
            f.write(' '.join(map(str, penalties_full)) + '\n')
            # Distance matrix (for subproblem)
            dist_matrix_int = np.round(sub_dist_matrix * scale).astype(int)
            for row in dist_matrix_int:
                f.write(' '.join(map(str, row)) + '\n')
                        
        # Call C++ Solver
        command = [
            SOLVER_PATH,
            tmp_file_path,
            str(min_prize_scaled),
            str(runs_per_instance)
        ]
            
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        output = result.stdout

        # Parse Output and Update Tour
        sub_tour = []
        for line in output.strip().split('\n'):
            if line.startswith("Best Result Route:"):
                full_route = [int(node) for node in line.split(':')[1].strip().split()]
                if len(full_route) > 2 and full_route[0] == 0 and full_route[-1] == 0:
                    sub_tour = full_route[1:-1]

        if not sub_tour:
            break
                
        unvisited_indices = np.where(unvisited_mask)[0]
        original_node_indices = unvisited_indices[sub_tour].tolist()
        
        # Append part of the new tour to the final tour based on the 'append' strategy
        if append_strategy == 'first':
            tour.append(original_node_indices[0])
        elif append_strategy == 'half':
            nodes_to_add = max(1, len(original_node_indices) // 2)
            tour.extend(original_node_indices[:nodes_to_add])
        else: # 'all'
            tour.extend(original_node_indices)
            
        # Clean up 
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    return tour