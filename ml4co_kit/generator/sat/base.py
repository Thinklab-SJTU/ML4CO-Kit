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
from enum import Enum
from typing import Union, List, Optional
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.task.sat.base import SATTaskBase


class SAT_DISTRIBUTION(str, Enum):
    """Define the SAT distribution types as an enumeration."""
    
    # Basic distributions
    UNIFORM_RANDOM = "uniform_random"  # Standard k-SAT with uniform random clauses
    
    # Theory-based distributions
    PHASE_TRANSITION = "phase_transition"  # Near satisfiability phase transition
    
    # Solution-based distributions
    PLANTED = "planted"  # Planted solution (guaranteed SAT)
    
    # Advanced distributions (G4SATBench)
    SR = "sr"  # Satisfiability Resolution (near SAT/UNSAT boundary)
    CA = "ca"  # Community Attachment (industrial-like)
    
    # NP-complete reductions
    K_CLIQUE = "k_clique"  # Reduction from k-clique problem
    K_DOMSET = "k_domset"  # Reduction from dominating set problem


class SATGeneratorBase(GeneratorBase):
    """Base class for all SAT problem generators."""
    
    def __init__(
        self, 
        task_type: TASK_TYPE,
        distribution_type: SAT_DISTRIBUTION = SAT_DISTRIBUTION.UNIFORM_RANDOM,
        precision: Union[np.float32, np.float64] = np.float32,
        vars_num: int = 50,
        clauses_num: Optional[int] = None,
        clause_length: int = 3,
        seed: Optional[int] = None,
    ):
        # Super Initialization
        super(SATGeneratorBase, self).__init__(
            task_type=task_type,
            distribution_type=distribution_type,
            precision=precision
        )
        
        # Initialize Attributes
        self.vars_num = vars_num
        self.clause_length = clause_length
        self.seed = seed
        
        # Auto-calculate clauses_num if not provided
        if clauses_num is None:
            self.clauses_num = self._default_clauses_num()
        else:
            self.clauses_num = clauses_num
        
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def _default_clauses_num(self) -> int:
        """Calculate default number of clauses based on clause length."""
        if self.clause_length == 2:
            # 2-SAT phase transition: α ≈ 1
            return int(1.0 * self.vars_num)
        elif self.clause_length == 3:
            # 3-SAT phase transition: α ≈ 4.26
            return int(4.26 * self.vars_num)
        else:
            # k-SAT phase transition: α ≈ 2^k * ln(2)
            alpha = (2 ** self.clause_length) * np.log(2)
            return int(alpha * self.vars_num)
    
    def _generate_random_clause(self, k: Optional[int] = None) -> List[int]:
        """Generate a random clause with k literals."""
        if k is None:
            k = self.clause_length
        
        # Randomly select k distinct variables
        selected_vars = np.random.choice(
            range(1, self.vars_num + 1), 
            size=min(k, self.vars_num), 
            replace=False
        )
        
        # Randomly assign polarities
        clause = []
        for var in selected_vars:
            if np.random.random() < 0.5:
                clause.append(int(var))  # Positive literal
            else:
                clause.append(-int(var))  # Negative literal
        
        return clause
    
    def _check_vig_connectivity(self, clauses: List[List[int]]) -> bool:
        """
        Check if Variable Incidence Graph (VIG) is connected.
        
        VIG: nodes are variables, edges connect variables appearing together.
        """
        try:
            import networkx as nx
        except ImportError:
            # If networkx not available, skip connectivity check
            return True
        
        # Build VIG
        G = nx.Graph()
        G.add_nodes_from(range(1, self.vars_num + 1))
        
        for clause in clauses:
            vars_in_clause = [abs(lit) for lit in clause]
            # Add edges between all pairs in this clause
            for i in range(len(vars_in_clause)):
                for j in range(i + 1, len(vars_in_clause)):
                    G.add_edge(vars_in_clause[i], vars_in_clause[j])
        
        return nx.is_connected(G)
    
    def _create_instance(self, clauses: List[List[int]], **kwargs) -> SATTaskBase:
        """
        Create a SAT task instance. 
        Subclasses should override this to return specific task types.
        """
        raise NotImplementedError("Subclasses should implement _create_instance")
