r"""
Graph Embedder.
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
from .utils import ScalarEmbeddingSine1D


class GraphEmbedder(GNN4COEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool):
        super(GraphEmbedder, self).__init__(hidden_dim, sparse)
        
        if self.sparse:
            # node embedder
            self.node_embed = nn.Sequential(
                ScalarEmbeddingSine1D(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
            # edge embedder
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine1D(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            
        else:
            raise ValueError("GraphEmbedder does not support dense input.")
            
    def sparse_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, 1); e: (E, 1)
        Return:
            x: (V, H); e: (E, H)
        """   
        x = self.node_embed(x.squeeze(1)) # (V, H)
        e = self.edge_embed(e.squeeze(1)) # (E, H)
        return x, e