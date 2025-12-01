r"""
ATSP Embedder.
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


from torch import Tensor, nn
from .base import GNN4COEmbedder
from .utils import ScalarEmbeddingSine1D, ScalarEmbeddingSine3D


class ATSPEmbedder(GNN4COEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool):
        super(ATSPEmbedder, self).__init__(hidden_dim, sparse)
        if self.sparse:
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine1D(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )   
        else:
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine3D(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )

    def sparse_forward(self, x: Tensor, e: Tensor) -> Tensor:
        """
        Args:
            x: (V, 2) [not use]
            e: (E,)
        Return:
            e: (E, H)
        """
        return self.edge_embed(e) # (E, H)
    
    def dense_forward(self, x: Tensor, e: Tensor) -> Tensor:
        """
        Args:
            x: (B, V, 2)  [not use]
            e: (B, V, V)
        Return:
            e: (B, V, V, H)
        """
        return self.edge_embed(e) # (B, V, V, H)