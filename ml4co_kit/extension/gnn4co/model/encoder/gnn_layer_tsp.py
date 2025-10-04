import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max


class GNNLayer(nn.Module):
    """Configurable GNN Layer
    Implements the Gated Graph ConvNet layer:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
        where Aggr. is an aggregation function: sum/mean/max.
    References:
        - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. 
          In International Conference on Learning Representations, 2018.
        - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. 
          arXiv preprint arXiv:2003.00982, 2020.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        aggregation: str = "sum", 
        norm: str = "batch", 
        learn_norm: bool = True, 
        track_norm: bool = False, 
        gated: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super(GNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)
        
    def forward(
        self, 
        h: torch.Tensor, 
        e: torch.Tensor, 
        graph: torch.Tensor, 
        mode: str="residual", 
        edge_index: torch.Tensor = None, 
        sparse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            In Dense version:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            mode: str
            In Sparse version:
            h: Input node features (V x H)
            e: Input edge features (E x H)
            graph: torch_sparse.SparseTensor
            mode: str
            edge_index: Edge indices (2 x E)
            sparse: Whether to use sparse tensors (True/False)
        Returns:
            Updated node and edge features
        """
        # torch.set_printoptions(threshold=numpy.inf)
        if not sparse:
            batch_size, num_nodes, hidden_dim = h.shape
        else:
            batch_size = None
            num_nodes, hidden_dim = h.shape

        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H

        if not sparse:
            Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H
        else:
            Vh = self.V(h[edge_index[1]])  # E x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H, source
        Bh = self.B(h)  # B x V x H, targetR
        Ce = self.C(e)  # B x V x V x H / E x H

        # Update edge features and compute edge gates
        if not sparse:
            e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        else:
            e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

        gates = torch.sigmoid(e)  # B x V x V x H / E x H

        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H

        # Normalize node features
        if not sparse:
            h = self.norm_h(
                h.view(batch_size * num_nodes, hidden_dim)
            ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
        else:
            h = self.norm_h(h) if self.norm_h else h

        # Normalize edge features
        if not sparse:
            e = self.norm_e(
                e.view(batch_size * num_nodes * num_nodes, hidden_dim)
            ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
        else:
            e = self.norm_e(e) if self.norm_e else e

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        if mode == "residual":
            h = h_in + h
            e = e_in + e

        return h, e
    
    def aggregate(
        self, 
        Vh: torch.Tensor, 
        graph: torch.Tensor, 
        gates: torch.Tensor, 
        mode: str = None, 
        edge_index: torch.Tensor = None, 
        sparse: bool = False
    ) -> torch.Tensor:
        """
        Args:
            In Dense version:
            Vh: Neighborhood features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            gates: Edge gates (B x V x V x H)
            mode: str
            In Sparse version:
            Vh: Neighborhood features (E x H)
            graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
            gates: Edge gates (E x H)
            mode: str
            edge_index: Edge indices (2 x E)
            sparse: Whether to use sparse tensors (True/False)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        # Enforce graph structure through masking
        # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

        # Aggregate neighborhood features
        if not sparse:
            if (mode or self.aggregation) == "mean":
                return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
            elif (mode or self.aggregation) == "max":
                return torch.max(Vh, dim=2)[0]
            else:
                return torch.sum(Vh, dim=2)
        else:
            sparseVh = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=Vh,
                sparse_sizes=(graph.size(0), graph.size(1))
            )

        if (mode or self.aggregation) == "mean":
            return sparse_mean(sparseVh, dim=1)
        elif (mode or self.aggregation) == "max": 
            return sparse_max(sparseVh, dim=1)
        else:
            return sparse_sum(sparseVh, dim=1)
