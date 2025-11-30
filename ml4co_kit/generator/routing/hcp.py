r"""
Generator for HCP instances.
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
from enum import Enum
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.routing.hcp import HCPTask
from ml4co_kit.generator.routing.base import RoutingGeneratorBase
from ml4co_kit.task.routing.base import DISTANCE_TYPE, ROUND_TYPE


class HCP_TYPE(str, Enum):
    """Define the HCP types as an enumeration."""
    RANDOM = "random"              # Random graph with specified edge probability
    HAMILTONIAN = "hamiltonian"    # Graph guaranteed to have Hamiltonian cycle
    NON_HAMILTONIAN = "non_hamiltonian"  # Graph guaranteed to have no Hamiltonian cycle


class HCPGenerator(RoutingGeneratorBase):
    """Generator for Hamiltonian Cycle Problem (HCP) instances."""
    
    def __init__(
        self, 
        distribution_type: HCP_TYPE = HCP_TYPE.RANDOM,
        precision: Union[np.float32, np.float64] = np.float32,
        nodes_num: int = 50,
        # special args for random
        edge_prob: float = 0.5,
        # special args for hamiltonian
        hamiltonian_min_degree: float = 0.6,  # Minimum degree ratio for Hamiltonian guarantee
        # special args for non_hamiltonian  
        non_hamiltonian_type: str = "bipartite",  # Type of non-Hamiltonian graph
    ):
        # Super Initialization
        super(HCPGenerator, self).__init__(
            task_type=TASK_TYPE.HCP, 
            distribution_type=distribution_type, 
            precision=precision
        )
        
        # Initialize Attributes
        self.nodes_num = nodes_num
        
        # Special Args for Different Distributions
        self.edge_prob = edge_prob
        self.hamiltonian_min_degree = hamiltonian_min_degree
        self.non_hamiltonian_type = non_hamiltonian_type
        
        # Generation Function Dictionary
        self.generate_func_dict = {
            HCP_TYPE.RANDOM: self._generate_random,
            HCP_TYPE.HAMILTONIAN: self._generate_hamiltonian,
            HCP_TYPE.NON_HAMILTONIAN: self._generate_non_hamiltonian,
        }
        
    def _create_hcp_instance(self, adj_matrix: np.ndarray) -> HCPTask:
        """Helper method to create HCP instance from adjacency matrix."""
        data = HCPTask(
            distance_type=DISTANCE_TYPE.ADJ_MATRIX,
            round_type=ROUND_TYPE.NO,
            precision=self.precision
        )
        data.from_data(adj_matrix=adj_matrix)
        return data
        
    def _generate_random(self) -> HCPTask:
        """Generate HCP instance with specified edge probability."""
        # Generate adjacency matrix with specified probability
        adj_matrix = np.random.binomial(
            1, self.edge_prob, size=(self.nodes_num, self.nodes_num)
        ).astype(int)
        
        # Ensure graph is undirected (symmetric)
        adj_matrix = np.triu(adj_matrix)  # Take upper triangle
        adj_matrix = adj_matrix + adj_matrix.T  # Make symmetric
        np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
        
        return self._create_hcp_instance(adj_matrix)

    def _generate_hamiltonian(self) -> HCPTask:
        """Generate HCP instance guaranteed to have Hamiltonian cycle."""
        # Start with a Hamiltonian cycle and add extra edges
        adj_matrix = np.zeros((self.nodes_num, self.nodes_num), dtype=int)
        
        # Create a base Hamiltonian cycle
        for i in range(self.nodes_num):
            adj_matrix[i, (i + 1) % self.nodes_num] = 1
            adj_matrix[(i + 1) % self.nodes_num, i] = 1
        
        # Add extra edges to ensure high connectivity
        min_edges = int(self.hamiltonian_min_degree * self.nodes_num)
        current_min_degree = 2  # From the base cycle
        
        while current_min_degree < min_edges:
            # Find nodes with minimum degree
            degrees = np.sum(adj_matrix, axis=1)
            min_degree_nodes = np.where(degrees == degrees.min())[0]
            
            # Add edges between low-degree nodes
            if len(min_degree_nodes) >= 2:
                i, j = np.random.choice(min_degree_nodes, size=2, replace=False)
                if adj_matrix[i, j] == 0 and i != j:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
            
            current_min_degree = np.min(degrees)
        
        # Add some random edges to increase variability
        extra_edges = int(0.1 * self.nodes_num * (self.nodes_num - 1) / 2)
        for _ in range(extra_edges):
            i, j = np.random.randint(0, self.nodes_num, size=2)
            if i != j and adj_matrix[i, j] == 0:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        return self._create_hcp_instance(adj_matrix)
    
    def _generate_non_hamiltonian(self) -> HCPTask:
        """Generate HCP instance guaranteed to have no Hamiltonian cycle."""
        if self.non_hamiltonian_type == "bipartite":
            return self._generate_bipartite_non_hamiltonian()
        elif self.non_hamiltonian_type == "cut_vertex":
            return self._generate_cut_vertex_non_hamiltonian()
        else:
            # Default to bipartite method
            return self._generate_bipartite_non_hamiltonian()
    
    def _generate_bipartite_non_hamiltonian(self) -> HCPTask:
        """Generate non-Hamiltonian graph using bipartite structure with unequal parts."""
        adj_matrix = np.zeros((self.nodes_num, self.nodes_num), dtype=int)
        
        # Create bipartite graph with unequal parts
        # This ensures no Hamiltonian cycle exists
        part1_size = self.nodes_num // 2 - 1  # Make parts unequal
        part2_size = self.nodes_num - part1_size
        
        # Connect nodes between parts (bipartite edges)
        for i in range(part1_size):
            for j in range(part1_size, self.nodes_num):
                # Add edge with high probability to make graph connected but non-Hamiltonian
                if np.random.random() < 0.7:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        # Add some intra-part edges to make it more challenging
        # But keep the graph bipartite in structure
        for i in range(part1_size):
            for j in range(i + 1, part1_size):
                if np.random.random() < 0.2:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        for i in range(part1_size, self.nodes_num):
            for j in range(i + 1, self.nodes_num):
                if np.random.random() < 0.2:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        return self._create_hcp_instance(adj_matrix)
    
    def _generate_cut_vertex_non_hamiltonian(self) -> HCPTask:
        """Generate non-Hamiltonian graph using cut vertex structure."""
        adj_matrix = np.zeros((self.nodes_num, self.nodes_num), dtype=int)
        
        # Create a graph with a cut vertex that separates components
        # This prevents Hamiltonian cycles
        cut_vertex = 0
        component1_size = self.nodes_num // 2
        component2_size = self.nodes_num - component1_size - 1
        
        # Build first component connected to cut vertex
        for i in range(1, component1_size + 1):
            # Connect to cut vertex
            adj_matrix[cut_vertex, i] = 1
            adj_matrix[i, cut_vertex] = 1
            
            # Add some edges within component
            for j in range(i + 1, component1_size + 1):
                if np.random.random() < 0.4:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        # Build second component connected to cut vertex
        for i in range(component1_size + 1, self.nodes_num):
            # Connect to cut vertex
            adj_matrix[cut_vertex, i] = 1
            adj_matrix[i, cut_vertex] = 1
            
            # Add some edges within component
            for j in range(i + 1, self.nodes_num):
                if np.random.random() < 0.4:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        # Ensure no direct connection between the two components
        # This creates a cut vertex at node 0
        
        return self._create_hcp_instance(adj_matrix)