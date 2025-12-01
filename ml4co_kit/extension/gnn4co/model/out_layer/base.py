from typing import Sequence
from torch import nn, Tensor


class OutLayerBase(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int, sparse: bool):
        super(OutLayerBase, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
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