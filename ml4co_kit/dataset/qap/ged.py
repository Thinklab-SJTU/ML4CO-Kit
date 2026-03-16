r"""
AIDS700nef Dataset for GED.
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


import os
import pathlib
import numpy as np
import networkx as nx
from typing import Union
from ml4co_kit.task.qap.ged import GEDTask
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.base import QAPGraphBase
from ml4co_kit.dataset.base import DatasetBase


class GEDAIDS700nefDataset(DatasetBase):
    """AIDS700nef Dataset for GED."""
    
    def __init__(
        self, 
        precision: Union[np.float32, np.float64] = np.float32,
    ):
        # Super Initialization  
        super(GEDAIDS700nefDataset, self).__init__(
            task_type=TASK_TYPE.GED,
            dataset_name="AIDS700nef",
            precision=precision
        )

        # Initialize Items
        self.items = [
            'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
            'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
            'Sb', 'Se', 'Ni', 'Te'
        ]
        self.len_items = len(self.items)

    def _preprocess(self):
        # Read all ``gexf`` files
        files = os.listdir(self.extracted_save_path)
        files.sort()
        self.cache["files"] = files
        
        # Generate all pairs for upper triangular matrix (excluding diagonal)
        # Pairs: (0,1), (0,2), ..., (0,699), (1,2), (1,3), ..., (698,699)
        pairs = []
        n = len(files)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        self.cache["pairs"] = pairs

    def _gexf_to_qap_graph(self, file_path: pathlib.Path) -> QAPGraphBase:
        # Read the gexf file into a NetworkX graph
        nx_graph: nx.Graph = nx.read_gexf(file_path)

        # Extract edge index
        edge_index = np.array(nx_graph.edges, dtype=np.int32).T

        # Extract node feature
        nodes_num = nx_graph.number_of_nodes()
        node_feature = np.zeros((nodes_num, self.len_items), dtype=self.precision)
        for node, info in nx_graph.nodes(data=True):
            node_feature[int(node), self.items.index(info['type'])] = 1

        # Default Edge Feature
        edges_num = edge_index.shape[1]
        edge_feature = np.ones((edges_num, 1), dtype=self.precision)

        # Create QAPGraphBase
        qap_graph = QAPGraphBase(precision=self.precision)
        qap_graph.from_data(
            node_feature=node_feature,
            edge_feature=edge_feature, 
            edge_index=edge_index
        )
        return qap_graph

    def _load(self, idx) -> GEDTask:
        # Get the graph pair indices
        g1_idx, g2_idx = self.cache["pairs"][idx]
        g1_file = self.cache["files"][g1_idx]
        g2_file = self.cache["files"][g2_idx]

        # Read the graph files into NetworkX graphs
        g1 = self._gexf_to_qap_graph(self.extracted_save_path / g1_file)
        g2 = self._gexf_to_qap_graph(self.extracted_save_path / g2_file)
        g1.make_symmetric()
        g2.make_symmetric()

        # Get the number of nodes
        n1 = g1.nodes_num
        n2 = g2.nodes_num
        if n1 > n2:
            graph1, graph2 = g2, g1
        else:
            graph1, graph2 = g1, g2

        # Create and return GMTask
        task_data = GEDTask(precision=self.precision)
        task_data.from_data(g1=graph1, g2=graph2)
        return task_data