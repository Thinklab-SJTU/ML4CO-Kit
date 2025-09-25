from torch import Tensor, nn
from ...model.embedder.base import GNN4COEmbedder
from ...model.embedder.utils import ScalarEmbeddingSine1D, ScalarEmbeddingSine3D


class ATSPEmbedder(GNN4COEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool):
        super(ATSPEmbedder, self).__init__(hidden_dim, sparse)
        if self.sparse:
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine1D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )   
        else:
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine3D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
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