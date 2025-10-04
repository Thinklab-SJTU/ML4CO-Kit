import math
import torch
from torch import nn, Tensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(
        self, 
        embedding_dim: int, 
        tau: int = 10000, 
        normalize: bool = False, 
        scale: float = None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, node_coords: Tensor) -> Tensor:
        """
        node_coords (B, N, 2)
        node_coords_x / node_coords_y (B, N)
        x_embed / y_embed (B, N, self.embedding_dim)
        embed (B, N, 2*self.embedding_dim)
        """
        
        # check dim of node_coords and get x and y of it
        dim_2 = False
        if node_coords.ndim == 2:
            dim_2 = True
            node_coords = node_coords.unsqueeze(0)
        node_coords_x = node_coords[:, :, 0]
        node_coords_y = node_coords[:, :, 1]
        
        # deal with normalize of node_coords_x/y
        if self.normalize:
            node_coords_x = node_coords_x * self.scale
            node_coords_y = node_coords_y * self.scale
            
        # get dim_t
        dim_t = torch.arange(
            self.embedding_dim, 
            dtype=torch.float32, 
            device=node_coords.device
        )
        dim_t = 2.0 * torch.div(dim_t, 2, rounding_mode='trunc') / self.embedding_dim
        dim_t = self.tau ** dim_t
        
        # (B, N) -> (B, N, self.embedding_dim)
        x_embed = node_coords_x[:, :, None] / dim_t
        y_embed = node_coords_y[:, :, None] / dim_t
        
        # sin for odd and cos for even
        x_embed_0_sin = x_embed[:, :, 0::2].sin()
        x_embed_0_cos = x_embed[:, :, 1::2].cos()
        x_embed = torch.stack((x_embed_0_sin, x_embed_0_cos), dim=3).flatten(2)
        y_embed_0_sin = y_embed[:, :, 0::2].sin()
        y_embed_0_cos = y_embed[:, :, 1::2].cos()
        y_embed = torch.stack((y_embed_0_sin, y_embed_0_cos), dim=3).flatten(2)
        
        # merge
        embed = torch.cat((x_embed, y_embed), dim=2).contiguous()
        
        # check dim
        if dim_2:
            embed = embed[0]
        return embed
    

class ScalarEmbeddingSine1D(nn.Module):
    def __init__(self, embedding_dim: int, tau: int=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tau = tau

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (V)
        embed: (V, self.embedding_dim)
        """
        # get dim_t
        dim_t = torch.arange(
            self.embedding_dim,
            dtype=torch.float32, 
            device=x.device
        )
        dim_t = 2.0 * torch.div(dim_t, 2, rounding_mode='trunc') / self.embedding_dim
        dim_t = self.tau ** dim_t
        
        # (N) -> (N, self.embedding_dim) 
        embed = x[:, None] / dim_t
        
        # sin for odd and cos for even
        embed_0_sin = embed[:, 0::2].sin()
        embed_1_cos = embed[:, 1::2].cos()
        embed = torch.stack((embed_0_sin, embed_1_cos), dim=2).flatten(1)
        
        return embed
    

class ScalarEmbeddingSine2D(nn.Module):
    def __init__(self, embedding_dim: int, tau: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tau = tau

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, V)
        embed: (B, V, self.embedding_dim)
        """
        # get dim_t
        dim_t = torch.arange(
            self.embedding_dim,
            dtype=torch.float32,
            device=x.device
        )
        dim_t = 2.0 * torch.div(dim_t, 2, rounding_mode='trunc') / self.embedding_dim
        dim_t = self.tau ** dim_t

        # (B, V) -> (B, V, self.embedding_dim)
        embed = x[:, :, None] / dim_t

        # sin for odd and cos for even
        embed_0_sin = embed[:, :, 0::2].sin()
        embed_1_cos = embed[:, :, 1::2].cos()
        embed = torch.stack((embed_0_sin, embed_1_cos), dim=3).flatten(2)

        return embed


class ScalarEmbeddingSine3D(nn.Module):
    def __init__(self, embedding_dim: int, tau: int=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tau = tau

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, V, V)
        embed: (B, V, V, self.embedding_dim)
        """
        # get dim_t
        dim_t = torch.arange(
            self.embedding_dim, 
            dtype=torch.float32, 
            device=x.device
        )
        dim_t = 2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.embedding_dim
        dim_t = self.tau ** dim_t
        
        # (B, H, W) -> (B, H, W, self.embedding_dim)
        embed = x[:, :, :, None] / dim_t
        
        # sin for odd and cos for even
        embed_0_sin = embed[:, :, :, 0::2].sin()
        embed_1_cos = embed[:, :, :, 1::2].cos()
        embed = torch.stack((embed_0_sin, embed_1_cos), dim=4).flatten(3)
        
        return embed