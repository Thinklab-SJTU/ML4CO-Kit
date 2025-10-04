from typing import Sequence
from torch import nn, Tensor


class GNN4COEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, sparse: bool):
        super(GNN4COEmbedder, self).__init__()
        
        # dims
        self.hidden_dim = hidden_dim

        # sparse
        self.sparse = sparse

    def forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            [sparse]
                x: (V,) or (V, C)
                e: (E,) or (E, C) 
            [dense]
                x: (B, V) or (B, V, C)
                e: (B, V, V) 
        Return:
            [sparse]
                x: (V, H)
                e: (E, H)
            [dense]
                x: (B, V, H) 
                e: (B, V, V, H) 
        """
        if self.sparse:
            return self.sparse_forward(x, e)
        else:
            return self.dense_forward(x, e)
        
    def sparse_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        raise NotImplementedError(
            "``sparse_forward`` is required to implemented in subclasses."
        )
    
    def dense_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        raise NotImplementedError(
            "``dense_forward`` is required to implemented in subclasses."
        )
    