r"""
ILS for SPCTSP
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


import os
import tempfile
import subprocess
import numpy as np
from ml4co_kit.task.routing.spctsp import SPCTSPTask
from ml4co_kit.solver.lib.ils.c_spctsp_ils import C_SPCTSP_ILS_SOLVER_PATH


def spctsp_ils(
    task_data: SPCTSPTask,
    ils_scale: int = 1e6,
    ils_runs: int = 1,
    spctsp_append_strategy: str = "half"
):
    # Preparation
    dists = task_data._get_dists()
    expected_prizes = task_data.expected_prizes
    actual_prizes = task_data.actual_prizes
    penalties = task_data.penalties
    required_prize = int(1.0 * ils_scale)
    spctsp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    
    # Before Loop
    tour = list()
    nodes_num = len(dists)
    remain_prize_to_collect = required_prize
    
    # Loop
    while remain_prize_to_collect > 0:
        # State Update
        visited_mask = np.zeros(nodes_num, dtype=bool)
        if tour:
            visited_mask[tour] = True
    
        # Sub-problem        
        current_dists = dists.copy()
        if tour:
            last_node_idx = tour[-1]
            current_dists[0, :] = dists[last_node_idx, :]
        unvisited_mask = ~visited_mask
        sub_dists = current_dists[np.ix_(unvisited_mask, unvisited_mask)]
        sub_penalties = penalties[unvisited_mask[1:]]
        sub_det_prizes = expected_prizes[unvisited_mask[1:]]
        
        # Check if the total collected stochastic prize is greater than 1.0
        total_collected_stoch_prize = np.sum(actual_prizes[np.array(tour) - 1]) if tour else 0.0
        if total_collected_stoch_prize >= 1.0:
            sol = np.array(tour)
            sol = np.insert(sol, 0, 0)
            sol = np.append(sol, 0)
            task_data.from_data(sol=sol, ref=False)
            return
        
        # Calculate the remaining prize to collect
        remain_prize_to_collect = (1.0 - total_collected_stoch_prize) * ils_scale
        min_prize_scaled = int(max(0, remain_prize_to_collect))
        
        # Calculate the maximum possible prize
        max_possible_prize_scaled = int(np.sum(sub_det_prizes) * ils_scale)
        min_prize_scaled = min(min_prize_scaled, max_possible_prize_scaled)
    
        # Write Subproblem to File
        sub_prizes = np.round(np.insert(sub_det_prizes, 0, 0) * ils_scale).astype(np.int32)
        sub_penalties = np.round(np.insert(sub_penalties, 0, 0) * ils_scale).astype(np.int32)
        sub_dists = np.round(sub_dists * ils_scale).astype(np.int32)
        with open(spctsp_file.name, 'w') as f:
            f.write(' '.join(map(str, sub_prizes)) + '\n')
            f.write(' '.join(map(str, sub_penalties)) + '\n')
            for row in sub_dists:
                f.write(' '.join(map(str, row)) + '\n')
            
        # Call the C++ solver with the scaled min_prize
        command = [
            C_SPCTSP_ILS_SOLVER_PATH,
            spctsp_file.name,
            str(min_prize_scaled), # Use the scaled integer value for min_prize
            str(ils_runs)
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding='utf-8'
        )
        output = result.stdout   
        
        # Parse Output
        sub_tour = []
        for line in output.strip().split('\n'):
            if line.startswith("Best Result Route:"):
                full_route = [int(node) for node in line.split(':')[1].strip().split()]
                if len(full_route) > 2 and full_route[0] == 0 and full_route[-1] == 0:
                    sub_tour = full_route[1:-1]
        if not sub_tour:
            break
        
        # Update Tour
        unvisited_indices = np.where(unvisited_mask)[0]
        original_node_indices = unvisited_indices[sub_tour].tolist()
        
        # Append part of the new tour to the final tour based on the 'append' strategy
        if spctsp_append_strategy == 'first':
            tour.append(original_node_indices[0])
        elif spctsp_append_strategy == 'half':
            nodes_to_add = max(1, len(original_node_indices) // 2)
            tour.extend(original_node_indices[:nodes_to_add])
        else: # 'all'
            tour.extend(original_node_indices)
        
    # Store the tour in the task_data
    sol = np.array(tour)
    sol = np.insert(sol, 0, 0)
    sol = np.append(sol, 0)
    task_data.from_data(sol=sol, ref=False)

    # Clean files
    os.remove(spctsp_file.name)