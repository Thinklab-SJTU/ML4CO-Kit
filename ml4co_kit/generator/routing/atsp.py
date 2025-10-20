r"""
Generator for ATSP instances.
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


import copy
import numpy as np
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.atsp import ATSPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class ATSP_TYPE(str, Enum):
    """Define the ATSP types as an enumeration."""
    UNIFORM = "uniform" # Uniform dists
    SAT = "sat" # SAT, transfer SAT to ATSP
    HCP = "hcp" # HCP, transfer HCP to ATSP


class ATSPGenerator(RoutingGeneratorBase):
    """Generator for ATSP instances."""
    
    def __init__(
        self, 
        distribution_type: ATSP_TYPE = ATSP_TYPE.UNIFORM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for sat
        sat_vars_nums: int = 4,
        sat_clauses_nums: int = 6,
    ):
        # Super Initialization
        super(ATSPGenerator, self).__init__(
            task_type=TASK_TYPE.ATSP, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = nodes_num
        
        # Special Args for SAT
        self.sat_vars_nums = sat_vars_nums
        self.sat_clauses_nums = sat_clauses_nums

        # Generation Function Dictionary
        self.generate_func_dict = {
            ATSP_TYPE.UNIFORM: self._generate_uniform,
            ATSP_TYPE.SAT: self._generate_sat,
            ATSP_TYPE.HCP: self._generate_hcp,
        }
        
    def _generate_uniform(self) -> ATSPTask:
        # Initialize dists
        dists = np.random.uniform(0.0, 1.0, size=(self.nodes_num, self.nodes_num))
        np.fill_diagonal(dists, 0)
        
        # Meet Triangle Constraint
        while True:
            old_dists = copy.deepcopy(dists)
            dists = (dists[:, None, :] + dists[None, :, :].transpose(0, 2, 1)).min(axis=2)
            if np.all(old_dists == dists):
                break
        
        # Create ATSP Instance from Data
        data = ATSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(dists=dists)
        return data

    def _generate_sat(self) -> ATSPTask:
        # Get number of variables and clauses
        num_variables = self.sat_vars_nums
        num_clauses = self.sat_clauses_nums
        
        # Calculate nodes_num
        nodes_num = 2 * num_clauses * num_variables + num_clauses
        
        # Initialize distance matrix with ones
        dists = np.ones((nodes_num, nodes_num))
        ref_tour = []
        
        # Randomly generate variable values (0 or 1)
        var_values = [np.random.randint(0, 2) for _ in range(num_variables)]
        
        # Generate the distance matrix and reference tour
        for v in range(num_variables):
            sub_tour = []
            ofs = v * 2 * num_clauses  # Offset for variable nodes
            
            for c in range(num_clauses):
                # Set the distances between clause pairs to 0
                dists[ofs + 2 * c, ofs + 2 * c + 1] = 0
                dists[ofs + 2 * c + 1, ofs + 2 * c] = 0
                
                # Build the sub-tour based on variable value
                if var_values[v] == 1:
                    sub_tour.append(ofs + 2 * c + 1)
                    if c != num_clauses - 1:
                        sub_tour.append(ofs + 2 * c + 2)
                else:
                    sub_tour.insert(0, ofs + 2 * c)
                    if c != num_clauses - 1:
                        sub_tour.insert(0, ofs + 2 * c + 1)
                    
                # Connect clauses in sequence
                if c != num_clauses - 1:
                    dists[ofs + 2 * c + 1, ofs + 2 * c + 2] = 0
                    dists[ofs + 2 * c + 2, ofs + 2 * c + 1] = 0

            # Connect variable clauses to the next variable
            dists[ofs, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
            dists[ofs, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0
            dists[ofs + 2 * num_clauses - 1, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
            dists[ofs + 2 * num_clauses - 1, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0

            # Extend the reference tour with the sub-tour
            ref_tour.extend(sub_tour)
            if var_values[(v + 1) % num_variables] == 1:
                ref_tour.append((ofs + 2 * num_clauses) % (2 * num_variables * num_clauses))
            else:
                ref_tour.append((ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses))

        # Close the tour by connecting the last node to the first
        ref_tour.insert(0, ref_tour[-1])
        ofs_clause = 2 * num_clauses * num_variables

        for c in range(num_clauses):
            # Randomly select 3 variables for the clause
            vars = np.random.choice(num_variables, size=3, replace=False)
            # Randomly assign signs (0 or 1) to the selected variables
            signs = np.random.choice(2, 3, replace=True)
            # Ensure at least one variable satisfies the clause
            fix_var_id = np.random.randint(0, 3)
            signs[fix_var_id] = var_values[vars[fix_var_id]]
            
            # Update distance matrix and reference tour for the clause
            for i in range(3):
                ofs_var = vars[i] * 2 * num_clauses
                if signs[i] == 1:  # Variable is True
                    dists[ofs_var + 2 * c, ofs_clause + c] = 0
                    dists[ofs_clause + c, ofs_var + 2 * c + 1] = 0
                    if vars[i] == vars[fix_var_id]:
                        ref_tour.insert(ref_tour.index(ofs_var + 2 * c + 1), ofs_clause + c)
                else:  # Variable is False (not x)
                    dists[ofs_var + 2 * c + 1, ofs_clause + c] = 0
                    dists[ofs_clause + c, ofs_var + 2 * c] = 0
                    if vars[i] == vars[fix_var_id]:
                        ref_tour.insert(ref_tour.index(ofs_var + 2 * c), ofs_clause + c)

        # Adjust the ref_tour to set the starting point to 0
        ref_tour = np.array(ref_tour)
        sol = np.roll(ref_tour[1:], -np.argmin(ref_tour[1:]))
        sol = np.insert(sol, len(sol), 0)
        
        # Create ATSP Instance from Data
        data = ATSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(dists=dists, sol=sol)
        return data
        
    def _generate_hcp(self) -> ATSPTask:
        # Preparation
        noise_level = np.random.rand() * 0.2 + 0.1
        num_nodes = self.nodes_num
        
        # Ones matrix
        dists = np.ones((num_nodes, num_nodes))

        # Random permutation of nodes (equivalent to torch.randperm)
        hpath = np.random.permutation(num_nodes)

        # Set distances to 0 along the hpath
        dists[hpath, np.roll(hpath, -1)] = 0

        # Add noise to the distance matrix
        num_noise_edges = int(noise_level * num_nodes * num_nodes)
        if num_noise_edges > 0:
            heads = np.random.choice(num_nodes, size=num_noise_edges, replace=True)
            tails = np.random.choice(num_nodes, size=num_noise_edges, replace=True)
            dists[heads, tails] = 0

        # Convert the hpath to a list and append the first node to form a closed tour
        ref_tour: list = hpath.tolist()
        ref_tour = np.array(ref_tour)
        sol = np.roll(ref_tour, -np.argmin(ref_tour))
        sol = np.insert(sol, len(sol), 0)
        
        # Create ATSP Instance from Data
        data = ATSPTask(
            distance_type=DISTANCE_TYPE.EUC_2D,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(dists=dists, sol=sol)
        return data