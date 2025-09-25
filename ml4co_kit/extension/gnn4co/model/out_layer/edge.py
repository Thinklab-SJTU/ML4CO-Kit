from typing import Sequence
from torch import Tensor, nn
from .utils import GroupNorm32
from .base import OutLayerBase


class EdgeOutLayer(OutLayerBase):
    def __init__(self, hidden_dim: int, out_channels: int, sparse: bool):
        super(EdgeOutLayer, self).__init__(hidden_dim, out_channels, sparse)
        self.e_norm = GroupNorm32(32, hidden_dim)
        self.e_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        
    def sparse_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, H); e: (E, H);
        Return:
            x: Any(not used); e: (E, out_channels);
        """
        nodes_num = x.shape[0]
        hidden_dim = e.shape[1]
        edges_num = e.shape[0]
        e = e.reshape(1, nodes_num, -1, hidden_dim).permute((0, 3, 1, 2))
        e: Tensor = self.e_out(self.e_norm(e))
        e = e.reshape(-1, edges_num).permute((1, 0))
        return x, e

    def dense_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (B, V, H); e: (B, V, V, H);
        Return:
            x: (B, out_channels, V); e: Any(not used);
        """
        e = self.e_out(self.e_norm(e.permute((0, 3, 1, 2)))) # (B, 2, V, V)
        return x, e