from typing import Sequence
from torch import nn, Tensor


class GNN4COEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, sparse: bool):
        super(GNN4COEmbedder, self).__init__()
        self.hidden_dim = hidden_dim
        self.sparse = sparse

    def forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
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
    