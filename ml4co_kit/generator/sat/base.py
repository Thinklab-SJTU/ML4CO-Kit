r"""
Base classes for all SAT problem generators.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
import pysat.solvers
import networkx as nx
from enum import Enum
from cnfgen import RandomKCNF
from itertools import combinations
from ml4co_kit.task.base import TASK_TYPE
from typing import Union, List, Optional, Tuple
from ml4co_kit.task.sat.base import SATTaskBase
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.generator.sat.c_ca_gen import pybind11_ca_gen_func
from ml4co_kit.generator.sat.c_ps_gen import pybind11_ps_gen_func


class SAT_TYPE(str, Enum):
    """Define the SAT types as an enumeration."""
    
    PHASE = "phase"  # Near satisfiability phase transition
    SR = "sr"  # SR Model
    CA = "ca"  # Community Attachment
    PS = "ps" # Popularity Similarity
    K_CLIQUE = "k_clique"  # Reduction from k-clique problem
    K_DOMSET = "k_domset"  # Reduction from dominating set problem
    K_VERCOV = "k_vercov"  # Reduction from vertex cover problem


class SATGeneratorBase(GeneratorBase):
    """Base class for all SAT problem generators."""
    
    def __init__(
        self, 
        task_type: TASK_TYPE,
        distribution_type: SAT_TYPE = SAT_TYPE.PHASE,
        precision: Union[np.float32, np.float64] = np.float32,
        # special args for phase
        phase_n_range: tuple = (10, 40),
        phase_k: int = 3,
        phase_alpha: float = 4.26,
        # special args for sr
        sr_n_range: tuple = (10, 40),
        sr_b: float = 0.3,
        sr_g: float = 0.4,
        # special args for ca
        ca_n_range: tuple = (10, 40),
        ca_mn_range: tuple = (13, 15),
        ca_k_range: tuple = (4, 5),
        ca_c_range: tuple = (3, 10),
        ca_q_range: tuple = (0.7, 0.9),
        # special args for ps
        ps_n_range: tuple = (10, 40),
        ps_mn_range: tuple = (6, 8),
        ps_k_range: tuple = (4, 5),
        ps_beta_range: tuple = (0.0, 1.0),
        ps_beta_prime: float = 1.0,
        ps_t_range: tuple = (0.75, 1.5),
        # special args for k_clique
        k_clique_v_range: tuple = (15, 20),
        k_clique_k_range: tuple = (3, 5),
        # special args for k_domset
        k_domset_v_range: tuple = (15, 20),
        k_domset_k_range: tuple = (3, 5),
        # special args for k_vercov
        k_vercov_v_range: tuple = (10, 20),
        k_vercov_k_range: tuple = (6, 8),
        # base solver
        base_solver: str = "cadical195"
    ):
        # Super Initialization
        super(SATGeneratorBase, self).__init__(
            task_type=task_type,
            distribution_type=distribution_type,
            precision=precision
        )
        
        # Special Args for SR
        self.sr_n_min, self.sr_n_max = sr_n_range
        self.sr_b = sr_b
        self.sr_g = sr_g
        
        # Special Args for Phase
        self.phase_n_min, self.phase_n_max = phase_n_range
        self.phase_k = phase_k
        self.phase_alpha = phase_alpha
        
        # Special Args for CA
        self.ca_n_min, self.ca_n_max = ca_n_range
        self.ca_mn_min, self.ca_mn_max = ca_mn_range
        self.ca_k_min, self.ca_k_max = ca_k_range
        self.ca_c_min, self.ca_c_max = ca_c_range
        self.ca_q_min, self.ca_q_max = ca_q_range
        
        # Special Args for PS
        self.ps_n_min, self.ps_n_max = ps_n_range
        self.ps_k_min, self.ps_k_max = ps_k_range
        self.ps_mn_min, self.ps_mn_max = ps_mn_range
        self.ps_beta_min, self.ps_beta_max = ps_beta_range
        self.ps_beta_prime = ps_beta_prime
        self.ps_t_min, self.ps_t_max = ps_t_range
        
        # Special Args for K-Clique
        self.k_clique_v_min, self.k_clique_v_max = k_clique_v_range
        self.k_clique_k_min, self.k_clique_k_max = k_clique_k_range
        
        # Special Args for K-Domset
        self.k_domset_v_min, self.k_domset_v_max = k_domset_v_range
        self.k_domset_k_min, self.k_domset_k_max = k_domset_k_range
        
        # Special Args for K-Vercov
        self.k_vercov_v_min, self.k_vercov_v_max = k_vercov_v_range
        self.k_vercov_k_min, self.k_vercov_k_max = k_vercov_k_range

        # Base Solver
        self.base_solver = base_solver
 
        # Generation Function Dictionary
        self.generate_func_dict = {
            SAT_TYPE.PHASE: self._generate_phase,
            SAT_TYPE.SR: self._generate_sr,
            SAT_TYPE.CA: self._generate_ca,
            SAT_TYPE.PS: self._generate_ps,
            SAT_TYPE.K_CLIQUE: self._generate_k_clique,
            SAT_TYPE.K_DOMSET: self._generate_k_domset,
            SAT_TYPE.K_VERCOV: self._generate_k_vercov,
        }

    def _generate_phase(self) -> SATTaskBase:
        raise NotImplementedError(
            "This function is required to be implemented in subclasses."
        )
    def _generate_sr(self) -> SATTaskBase:
        raise NotImplementedError(
            "This function is required to be implemented in subclasses."
        )

    def _generate_ca(self) -> SATTaskBase:
        raise NotImplementedError(
            "This function is required to be implemented in subclasses."
        )

    def _generate_ps(self) -> SATTaskBase:
        raise NotImplementedError(
            "This function is required to be implemented in subclasses."
        )

    def _generate_k_clique(self) -> SATTaskBase:
        raise NotImplementedError(
            "This function is required to be implemented in subclasses."
        )

    def _generate_k_domset(self) -> SATTaskBase:
        raise NotImplementedError(
            "This function is required to be implemented in subclasses."
        )

    def _generate_k_vercov(self) -> SATTaskBase:
        raise NotImplementedError(
            "This function is required to be implemented in subclasses."
        )

    def _super_generate_phase(self) -> List[List[int]]:
        """
        @article{
            crawford1996experimental,
            title={Experimental results on the crossover point in random 3-SAT},
            author={Crawford, James M and Auton, Larry D},
            journal={Artificial intelligence},
            volume={81},
            number={1-2},
            pages={31--57},
            year={1996},
            publisher={Elsevier}
        }
        """
        # Randomly sample number of variables
        num_vars = np.random.randint(self.phase_n_min, self.phase_n_max + 1)

        # Generate clauses
        while True:
            # Randomly sample number of clauses
            num_clauses = int(self.phase_alpha * num_vars)

            # Call `RandomKCNF` to generate a random k-CNF formula
            cnf = RandomKCNF(k=self.phase_k, n=num_vars, m=num_clauses)

            # Get clauses
            clauses = list(cnf.clauses())
            
            # Ensure the graph in connected
            if self._check_vig(num_vars, clauses):
                break
        
        # Remove duplicate unsat clauses and sat clauses
        clauses = self._clean_clauses(clauses)
        return clauses

    def _super_generate_sr(self) -> Tuple[List[List[int]], bool, Optional[List[int]]]:
        """
        @article{
            selsam2018learning,
            title={Learning a SAT solver from single-bit supervision},
            author={Selsam, Daniel and Lamm, Matthew and B{\"u}nz, Benedikt \
                and Liang, Percy and de Moura, Leonardo and Dill, David L},
            journal={arXiv preprint arXiv:1802.03685},
            year={2018}
        }
        """
        while True:
            # Get base solver
            solver = pysat.solvers.Solver(self.base_solver)

            # Randomly sample number of variables
            num_vars = np.random.randint(self.sr_n_min, self.sr_n_max + 1)

            # Generate clauses
            clauses = []
            sat_sol = None
            while True:
                # Randomly choose k = 1 + Bernoulli(b) + Geometric(g)
                term_2 = np.random.random() < self.sr_b
                term_3 = np.random.geometric(self.sr_g)
                k = 1 + term_2 + term_3

                # Randomly choose k variables without replacement
                vs = np.random.choice(num_vars, size=min(num_vars, k), replace=False)
                clause = [int(v + 1) if np.random.random() < 0.5 else int(-(v + 1)) for v in vs]

                # Add clause to solver
                solver.add_clause(clause)
                
                # If clause is satisfiable, add to clauses, otherwise break
                if solver.solve():
                    clauses.append(clause)
                    sat_sol = solver.get_model()
                else:
                    break

            # Unsat clauses and sat clauses
            unsat_clause = clause
            sat_clause = [-clause[0]] + clause[1:]
            unsat_clauses = clauses + [unsat_clause]
            sat_clauses = clauses + [sat_clause]

            # Ensure the graph in connected
            if self._check_vig(num_vars, unsat_clauses):
                break

        # Remove duplicate unsat clauses and sat clauses
        unsat_clauses = self._clean_clauses(unsat_clauses)
        sat_clauses = self._clean_clauses(sat_clauses)

        # Return unsat clauses, sat clauses and sat solution
        return unsat_clauses, sat_clauses, sat_sol

    def _super_generate_ca(self):
        """
        CA: https://github.com/jgirald/communityAttachment
        @article{
            giraldez2015modularity,
            title={A modularity-based random SAT instances generator},
            author={Gir{\'a}ldez-Cru, Jes{\'u}s and Levy, Jordi},
            year={2015},
            publisher={AAAI Press}
        }
        """
        # Randomly sample number of literals per clause
        k = np.random.randint(self.ca_k_min, self.ca_k_max + 1)

        # Randomly sample number of variables
        num_vars = np.random.randint(self.ca_n_min, self.ca_n_max + 1)
        while max(self.ca_k_min, k) > min(self.ca_c_max, int(num_vars / k)):
            num_vars = np.random.randint(self.ca_n_min, self.ca_n_max + 1)

        # Randomly sample number of clauses
        mn_ratio = np.random.uniform(self.ca_mn_min, self.ca_mn_max)
        num_clauses = int(mn_ratio * num_vars)

        # Randomly sample modularity parameter
        q = np.random.uniform(self.ca_q_min, self.ca_q_max)
        
        # According to Giráldez-Cru and Levy's original paper and official implementation, 
        # the CA generator requires the number of communities c to satisfy k ≤ c ≤ n/k.
        # This means there must be at least k communities, and each community should be assigned at least k variables.
        # min(self.ca_c_max, int(num_vars / k)) ensures that c does not become too large to violate the constraints.
        # max(self.ca_c_min, k) ensures there are at least k communities, as required by the CA model.
        c = np.random.randint(max(self.ca_c_min, k), min(self.ca_c_max, int(num_vars / k)) + 1)

        # Randomly sample seed
        seed = np.random.randint(0, 2**16)

        # Call PyBind11 CA generator to generate CA clauses
        clauses = pybind11_ca_gen_func(
            int(num_vars),      # Number of variables
            int(num_clauses),   # Number of clauses
            int(k),             # Number of literals per clause
            float(q),           # Modularity parameter
            int(c),             # Number of communities
            int(seed)           # Seed
        )
        return clauses
    
    def _super_generate_ps(self):
        """
        PS: https://github.com/jgirald/ps-sat
        @inproceedings{
            giraldez2017locality,
            title={Locality in random SAT instances},
            author={Gir{\'a}ldez-Cru, Jes{\'u}s and Levy, Jordi},
            year={2017},
            organization={International Joint Conferences on Artificial Intelligence}
        }
        """
        # Randomly sample number of literals per clause
        k = np.random.randint(self.ps_k_min, self.ps_k_max + 1)

        # Randomly sample number of variables
        num_vars = np.random.randint(self.ps_n_min, self.ps_n_max + 1)

        # Randomly sample number of clauses
        mn_ratio = np.random.uniform(self.ps_mn_min, self.ps_mn_max)
        num_clauses = int(mn_ratio * num_vars)

        # Randomly sample beta
        beta = np.random.uniform(self.ps_beta_min, self.ps_beta_max)

        # Randomly sample t
        t = np.random.uniform(self.ps_beta_min, self.ps_beta_max)

        # Randomly sample seed
        seed = np.random.randint(0, 2**16)
        
        # Call PyBind11 PS generator to generate PS clauses
        # Convert numpy types to Python native types for pybind11 compatibility
        clauses = pybind11_ps_gen_func(
            int(num_vars),              # Number of variables
            int(num_clauses),           # Number of clauses
            int(k),                     # Average clause size
            int(k-3),                   # Rigid clause size
            float(beta),                # Beta
            float(self.ps_beta_prime),  # Beta prime
            float(t),                   # Temperature
            int(seed)                   # Seed
        )
        return clauses

    def _super_generate_k_clique(self):
        pass

    def _super_generate_k_domset(self):
        pass

    def _super_generate_k_vercov(self):
        pass

    def _check_vig(self, num_vars: int, clauses: List[List[int]]) -> bool:
        # Create networkx graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(num_vars))

        # Add edges between all pairs in each clause
        for clause in clauses:
            v_idxs = [abs(literal) - 1 for literal in clause]
            edges = list(combinations(v_idxs, 2))
            nx_graph.add_edges_from(edges)

        # Return if the graph is connected
        return nx.is_connected(nx_graph)

    def _clean_clauses(self, clauses: List[List[int]]) -> List[List[int]]:
        hash_clauses = []
        cleaned_clauses = []
        for clause in clauses:
            hash_clause = hash(frozenset([str(literal).encode() for literal in clause]))
            if hash_clause in hash_clauses:
                continue
            hash_clauses.append(hash_clause)
            cleaned_clauses.append(clause)
        return cleaned_clauses