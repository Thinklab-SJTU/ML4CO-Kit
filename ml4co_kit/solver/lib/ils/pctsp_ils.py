r"""
ILS for PCTSP
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
from ml4co_kit.task.routing.pctsp import PCTSPTask
from ml4co_kit.solver.lib.ils.c_pctsp_ils import C_PCTSP_ILS_SOLVER_PATH


def pctsp_ils(
    task_data: PCTSPTask,
    ils_scale: int = 1e6,
    ils_runs: int = 1
):
    # Preparation
    dists = np.round(task_data._get_dists() * ils_scale).astype(np.int32)
    prizes: np.ndarray = np.insert(task_data.norm_prizes, 0, 0)
    prizes = np.round(prizes * ils_scale).astype(np.int32)
    penalties: np.ndarray = np.insert(task_data.penalties, 0, 0)
    penalties = np.round(penalties * ils_scale).astype(np.int32)
    required_prize = int(1.0 * ils_scale)
    pctsp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    
    # Write the properly scaled integer data to a tmporary file
    with open(pctsp_file.name, 'w') as f:
        # Prizes
        f.write(' '.join(map(str, prizes)) + '\n')
        # penalties
        f.write(' '.join(map(str, penalties)) + '\n')
        # Distance matrix
        for row in dists:
            f.write(' '.join(map(str, row)) + '\n')
            
    # Call the C++ solver with the scaled min_prize
    command = [
        C_PCTSP_ILS_SOLVER_PATH,
        pctsp_file.name,
        str(required_prize), # Use the scaled integer value for min_prize
        str(ils_runs)
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, encoding='utf-8'
    )
    output = result.stdout   

    # Process the output of the C++ solver
    tour = None
    for line in output.strip().split('\n'):
        if line.startswith("Best Result Route:"):
            full_route = [int(node) for node in line.split(':')[1].strip().split()]
            if len(full_route) > 2 and full_route[0] == 0 and full_route[-1] == 0:
                tour = np.array(full_route)
            else:
                raise RuntimeError("Failed to solve a route from C++ solver output.")
    if tour is None:
        raise RuntimeError("Failed to solve a route from C++ solver output.")
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)

    # Clean files
    os.remove(pctsp_file.name)