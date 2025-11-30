r"""
TSP Embedder.
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


from typing import Sequence
from torch import Tensor, nn
from .base import GNN4COEmbedder
from .utils import (
    PositionEmbeddingSine, ScalarEmbeddingSine1D, ScalarEmbeddingSine3D
)


class TSPEmbedder(GNN4COEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool):
        super(TSPEmbedder, self).__init__(hidden_dim, sparse)
        
        if self.sparse:
            # node embedder
            self.node_embed = nn.Sequential(
                PositionEmbeddingSine(self.hidden_dim // 2),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
            # edge embedder
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine1D(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            
        else:
            # node embedder
            self.node_embed = nn.Sequential(
                PositionEmbeddingSine(self.hidden_dim // 2),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
            # edge embedder
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine3D(hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )

    def sparse_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, 2) nodes_feature (node coords)
            e: (E,) edges_feature (distance matrix)
        Return:
            x: (V, H)
            e: (E, H)
        """   
        x = self.node_embed(x) # (V, H)
        e = self.edge_embed(e) # (E, H)
        return x, e
    
    def dense_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (B, V, 2) nodes_feature (node coords)
            e: (B, V, V) edges_feature (distance matrix)
        Return:
            x: (B, V, H)
            e: (B, V, V, H)
        """
        x = self.node_embed(x) # (B, V, H)
        e = self.edge_embed(e) # (B, V, V, H)
        return x, e