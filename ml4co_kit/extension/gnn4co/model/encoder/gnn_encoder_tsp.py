import torch
from typing import Sequence
from torch import nn, Tensor
from torch_sparse import SparseTensor
from ..encoder.gnn_layer_tsp import GNNLayer
from ..embedder.utils import PositionEmbeddingSine, ScalarEmbeddingSine3D


class TSPGNNEncoder(nn.Module):
    def __init__(
        self, 
        sparse: bool,
        num_layers: int = 12,
        hidden_dim: int = 256, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False
    ):
        super(TSPGNNEncoder, self).__init__()
        
        # info
        self.sparse = sparse
        self.hidden_dim = hidden_dim
        self.time_embed_dim = hidden_dim // 2
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        
        # embedder
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_pos_embed = ScalarEmbeddingSine3D(hidden_dim)
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

        # out layer
        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True)
        )
        
        # gnn layers
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm)
            for _ in range(num_layers)
        ])
            
        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim)
            ),
            ) for _ in range(num_layers)
        ])
    
    def forward(
        self, task: str, x: Tensor, e: Tensor, edge_index: Tensor
    ) -> Sequence[Tensor]:
        # dense
        if self.sparse:
            # node embedder
            x: Tensor = self.pos_embed(x.unsqueeze(0)) # (1, V, H)
            x = self.node_embed(x.squeeze(0)) # (V, H)
            
            # edge embedder
            e: Tensor = self.edge_pos_embed(e.expand(1, 1, -1)) # (1, E, H)
            e = self.edge_embed(e.squeeze()) # (E, H)

            # gnn layer
            edge_index = edge_index.long()
            adj_matrix: Tensor = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=torch.ones_like(edge_index[0].float()),
                sparse_sizes=(x.shape[0], x.shape[0]),
            )
            adj_matrix = adj_matrix.to(x.device)
            for layer, out_layer in zip(self.layers, self.per_layer_out):
                x_in, e_in = x, e
                x, e = layer(x_in, e_in, adj_matrix, edge_index=edge_index, sparse=True, mode="direct")
                x = x_in + x
                e = e_in + out_layer(e)
            
            # out
            e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
            e = self.out(e)
            e = e.reshape(-1, edge_index.shape[1]).permute((1, 0))
        
        else:
            # embedder
            graph = torch.ones_like(e).long()
            import pdb
            x = self.node_embed(self.pos_embed(x)) # (B, V, H)
            e = self.edge_embed(self.edge_pos_embed(e)) # (B, V, V, H)
            
            # gnn layer
            for layer, out_layer in zip(self.layers, self.per_layer_out):
                x_in, e_in = x, e
                x, e = layer(x, e, graph, mode="direct")
                x = x_in + x
                e = e_in + out_layer(e)
            
            # out layer
            e = self.out(e.permute((0, 3, 1, 2)))

        # return
        return x, e    


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).type(x.dtype)
    

def normalization(channels: int):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module