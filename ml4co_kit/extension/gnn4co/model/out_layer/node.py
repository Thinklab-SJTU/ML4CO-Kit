from typing import Sequence
from torch import Tensor, nn
from .utils import GroupNorm32
from .base import OutLayerBase


class NodeOutLayer(OutLayerBase):
    def __init__(self, hidden_dim: int, out_channels: int, sparse: bool):
        super(NodeOutLayer, self).__init__(hidden_dim, out_channels, sparse)
        self.x_norm = GroupNorm32(32, hidden_dim)
        if self.sparse:
            self.x_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        else:
            self.x_out = nn.Linear(hidden_dim, out_channels, bias=True)

    def sparse_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, H); e: (E, H);
        Return:
            x: (V, out_channels); e: Any(not used);
        """
        nodes_num, hidden_dim = x.shape
        x = x.reshape(1, nodes_num, -1, hidden_dim).permute((0, 3, 1, 2))
        x: Tensor = self.x_out(self.x_norm(x))
        x = x.reshape(-1, nodes_num).permute((1, 0))
        return x, e
    
    def dense_forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (B, V, H); e: (B, V, V, H);
        Return:
            x: (B, out_channels, V); e: Any(not used);
        """
        x = self.x_norm(x.permute(0, 2, 1)) # (B, H, V)
        x = x.permute(0, 2, 1) # (B, V, H)
        x = self.x_out(x) # (B, V, out_channels)
        x = x.permute(0, 2, 1) # (B, out_channels, V)
        return x, e