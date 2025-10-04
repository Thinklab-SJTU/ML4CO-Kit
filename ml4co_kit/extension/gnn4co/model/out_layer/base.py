from typing import Sequence
from torch import nn, Tensor


class OutLayerBase(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int, sparse: bool):
        super(OutLayerBase, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.sparse = sparse
        
    def forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            [sparse]
                x: (V, H)
                e: (E, H)
                t: (H)
            [dense]
                x: (B, V, H) 
                e: (B, V, V, H) 
                t: (H) 
        Return:
            [sparse]
                x: (V, out_channels)
                e: (E, out_channels)
                t: (H)
            [dense]
                x: (B, out_channels, V) 
                e: (B, out_channels, V, V) 
                t: (H) 
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