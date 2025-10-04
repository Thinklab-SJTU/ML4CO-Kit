import torch
import torch.nn.functional as F
from typing import Sequence
from torch import Tensor, nn
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max


class GNNSparseLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        aggregation: str = "sum", 
        norm: str = "batch",
        learn_norm: bool = True,
        track_norm: bool = False,
        asym: bool = False
    ):
        super(GNNSparseLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Linear Layer for nodes
        if not asym:
            self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Linear Layer for edges
        if asym:
            self.D = nn.Linear(2, hidden_dim, bias=True)
            self.F = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
            self.E1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.E2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        else:
            self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Normalization for nodes and edges
        if norm == "batch":
            self.norm_x = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
            self.norm_e = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        else:
            self.norm_x = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
            self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)

    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, H) Node features; e: (E, H) Edge features
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes.
        Returns:
            Updated x and e after one layer of GNN.
        """
        nodes_num = x.shape[0] # Total number of nodes
        
        # Linear transformation for node embeddings
        Ux: Tensor = self.U(x) # (V, H)
        
        # Aggregate neighbor information for edges
        Vx = self.V(x[edge_index[1]]) # (E, H)
        
        # Message passing from nodes to edges
        Ax = self.A(x) # (V, H), source
        Bx = self.B(x) # (V, H), target
        
        # Update edge features
        Ce = self.C(e) # (E, H)
        e = Ax[edge_index[0]] + Bx[edge_index[1]] + Ce # (E, H)
            
        # Sigmoid gates for edge features
        gates = torch.sigmoid(e) # (E, H)
        
        # Aggregate messages for node embeddings
        x = Ux + self.aggregate(Vx, gates, edge_index, nodes_num) # (V, H)

        # Apply normalization and activation
        x = F.relu(self.norm_x(x)) # (V, H)
        e = F.relu(self.norm_e(e)) # (E, H)
        
        return x, e
    
    def asym_forward(
        self, e: Tensor, edges_feature: Tensor, edge_index: Tensor, nodes_num: int
    ) -> Sequence[Tensor]:
        """
        Args:
            e: (E, H) Decision Variables (edge)
            edges_feature: (E, 2) Edge features (distance matrix)
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes.
        Returns:
            Updated x and e after one layer of GNN.
        """
        edges_num = e.shape[0] # Total number of edges
        
        # Update edge features
        Ce = self.C(e) # (E, H)
        De = self.D(edges_feature) # (E, H)
        E1e = self.E1(e) # (E, H) [source]
        E2e: Tensor = self.E2(e) # (E, H) [target]
        
        # Sigmoid gates for edge features
        gates = torch.sigmoid(e) # (B, V, V, H)

        # Aggregate messages for node embeddings
        edge_index_inv = torch.stack([edge_index[1], edge_index[0]], dim=0)
        E1e = self.aggregate(E1e, gates, edge_index, nodes_num) # (V, H)
        E2e = self.aggregate(E2e, gates, edge_index_inv, nodes_num) # (V, H)
        
        # Update edge features
        E_aggre = torch.cat([E1e.unsqueeze(1).repeat(1, nodes_num, 1),
                              E2e.unsqueeze(0).repeat(nodes_num, 1, 1)], dim=2) # (V, V, 2H)
        E_aggre = E_aggre[edge_index[0], edge_index[1]] # (E, 2H)
        e = self.F(E_aggre) + Ce + De # (E, H)
        
        # Apply normalization and activation
        e = F.relu(self.norm_e(e)) # (E, H)
        
        return e
    
    
    def aggregate(
        self, Vx: Tensor, gates: Tensor, edge_index: Tensor, nodes_num: int
    ) -> Tensor:
        """
        Args:
            Vx: (E, H); gates: (E, H); edge_index: (2, E)

        Returns:
            node feature: (V, H)
        """
        sparseVh = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=Vx * gates,
            sparse_sizes=(nodes_num, nodes_num)
        )
        if self.aggregation == "mean":
            return sparse_mean(sparseVh, dim=1)
        elif self.aggregation == "max":
            return sparse_max(sparseVh, dim=1)
        else:
            return sparse_sum(sparseVh, dim=1)
        
        
class GNNSparseBlock(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        hidden_dim: int, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False,
        asym: bool = False
    ):
        super(GNNSparseBlock, self).__init__()
        
        # gnn layer
        self.layers = nn.ModuleList([
            GNNSparseLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, asym)
            for _ in range(num_layers)
        ])
        
        # per layer out
        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
                nn.SiLU(),
                zero_module(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, H) Node features; 
            e: (E, H) Edge features;
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes.
        
        Return:
            updated features. x: (V, H); e: (E, H);
        """
        # gnn layer
        for layer, out_layer in zip(self.layers, self.per_layer_out):
            x_in, e_in = x, e
            x, e = layer(x, e, edge_index)
            x = x + x_in
            e = e_in + out_layer(e)
        
        # return
        return x, e

    def asym_forward(
        self, e: Tensor, edges_feature: Tensor, edge_index: Tensor, nodes_num: int
    ) -> Sequence[Tensor]:
        """
        Args:
            e: (E, H) Decision Variables (edge);
            edges_feature: (E, H) Edge features;
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes.
            
        Return:
            updated feature e: (E, H)
        """
        # gnn layer
        for layer, out_layer in zip(self.layers, self.per_layer_out):
            layer: GNNSparseLayer
            e_in = e
            e = layer.asym_forward(e, edges_feature, edge_index, nodes_num)
            e = e_in + out_layer(e)
        
        # return
        return e
    

class GNNDenseLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        aggregation: str = "sum", 
        norm: str = "batch",
        learn_norm: bool = True,
        track_norm: bool = False,
        asym: bool = False
    ):
        super(GNNDenseLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Linear Layer for nodes
        if not asym:
            self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Linear Layer for edges
        if asym:
            self.D = nn.Linear(2, hidden_dim, bias=True)
            self.F = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
            self.E1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.E2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        else:
            self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Normalization for nodes and edges
        if asym:
            if norm == "batch":
                self.norm_x1 = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
                self.norm_x2 = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
                self.norm_e = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
            else:
                self.norm_x1 = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
                self.norm_x2 = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
                self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
        else:
            if norm == "batch":
                self.norm_x = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
                self.norm_e = nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
            else:
                self.norm_x = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
                self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm)
                
    def forward(self, x: Tensor, e: Tensor, graph: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (B, V, H) Node features; 
            e: (B, V, V, H) Edge features
            graph: (B, V, V) Graph adjacency matrices
        Returns:
            Updated x and e after one layer of GNN.
        """
        batch_size, nodes_num, hidden_dim = x.shape
        
        # Linear transformation for node embeddings
        Ux: Tensor = self.U(x) # (B, V, H)
        
        # Aggregate neighbor information for edges
        Vx: Tensor = self.V(x) # (B, V, H)
        Vx = Vx.unsqueeze(1).expand(-1, nodes_num, -1, -1) # (B, V, V, H)
        
        # Message passing from nodes to edges
        Ax: Tensor = self.A(x) # (B, V, H), source
        Bx: Tensor = self.B(x) # (B, V, H), target
        
        # Update edge features
        Ce = self.C(e) # (B, V, V, H)
        e = Ax.unsqueeze(dim=1) + Bx.unsqueeze(dim=2) + Ce # (B, V, V, H)
            
        # Sigmoid gates for edge features
        gates = torch.sigmoid(e) # (B, V, V, H)
        
        # Aggregate messages for node embeddings
        x = Ux + self.aggregate(Vx, gates, graph) # (B, V, H)

        # Apply normalization and activation
        x = x.view(batch_size * nodes_num, hidden_dim) # (B*V, H)
        x = F.relu(self.norm_x(x)).view(batch_size, nodes_num, hidden_dim) # (B, V, H)
        e = e.view(batch_size * nodes_num * nodes_num, hidden_dim)
        e = F.relu(self.norm_e(e)).view(batch_size, nodes_num, nodes_num, hidden_dim) # (B, V, V, H)
        
        return x, e
    
    def asym_forward(self, e: Tensor, edges_feature: Tensor, graph: Tensor) -> Sequence[Tensor]:
        """
        Args:
            e: (B, V, V, H) Decision Variables (edge)
            edges_feature: (B, V, V) Edge features (distance martix)
            graph: (B, V, V) Graph adjacency matrices
        Returns:
            Updated x and e after one layer of GNN.
        """
        batch_size, nodes_num, _, hidden_dim = e.shape
        
        # Update edge features
        Ce = self.C(e) # (B, V, V, H)
        edges_feature = torch.stack([edges_feature, edges_feature.transpose(1, 2)], dim=3) # (B, V, V, 2)
        De = self.D(edges_feature) # (B, V, V, H)
        E1e = self.E1(e) # (B, V, V, H) [source]
        E2e: Tensor = self.E2(e) # (B, V, V, H) [target]
        E2e = E2e.transpose(1, 2) # (B, V, V, H) [target]
        
        # Sigmoid gates for edge features
        gates = torch.sigmoid(e) # (B, V, V, H)
        gates_T = torch.sigmoid(e.transpose(1, 2))

        # Aggregate messages for node embeddings
        E1e = self.aggregate(E1e, gates, graph) # (B, V, H)
        E2e = self.aggregate(E2e, gates_T, graph) # (B, V, H)
        
        # Update edge features
        E_aggre = torch.cat([E1e.unsqueeze(2).repeat(1, 1, nodes_num, 1),
                              E2e.unsqueeze(1).repeat(1, nodes_num, 1, 1)], dim=3) # (B, V, V, 2H)
        e = self.F(E_aggre) + Ce + De # (B, V, V, H)
        
        # Apply normalization and activation
        e = e.view(batch_size * nodes_num * nodes_num, hidden_dim) # (B*V*V, H)
        e = F.relu(self.norm_e(e)).view(batch_size, nodes_num, nodes_num, hidden_dim) # (B, V, V, H)
        
        return e
    
    def aggregate(self, Vx: Tensor, gates: Tensor, graph: Tensor) -> Tensor:  
        """
        Args:
            Vx: (B, V, H); gates: (B, V, V, H); graph: (B, V, V)

        Returns:
            node feature: (B, V, H)
        """
        Vx = Vx * gates
        if self.aggregation == "mean":
            return torch.sum(Vx, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vx))
        elif self.aggregation == "max":
            return torch.max(Vx, dim=2)[0]
        else:
            return torch.sum(Vx, dim=2)
        
class GNNDenseBlock(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        hidden_dim: int, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False,
        asym: bool = False
    ):
        super(GNNDenseBlock, self).__init__()
        
        # gnn layer
        self.layers = nn.ModuleList([
            GNNDenseLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, asym)
            for _ in range(num_layers)
        ])
        
        # per layer out
        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
                nn.SiLU(),
                zero_module(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x: Tensor, e: Tensor, edge_index) -> Sequence[Tensor]:
        """
        Args:
            x: (B, V, H) Node features; 
            e: (B, V, V, H) Edge features;
            edge_index: None
            
        Return:
            updated features. x: (B, V, H); e: (B, V, V, H);
        """
        batch_size, nodes_num, _ = x.shape
        graph = torch.ones(size=(batch_size, nodes_num, nodes_num)).to(x.device)
        
        # gnn layer
        for layer, out_layer in zip(self.layers, self.per_layer_out):
            x_in, e_in = x, e
            x, e = layer(x, e, graph)
            x = x + x_in
            e = e_in + out_layer(e)
        
        # return
        return x, e

    def asym_forward(
        self, e: Tensor, edges_feature: Tensor, edge_index: Tensor, nodes_num: int 
    ) -> Sequence[Tensor]:
        """
        Args:
            e: (B, V, V, H) Decision Variables (edge);
            edges_feature: (B, V, V, H) Edge features;
            edge_index: None
            
        Return:
            updated features. x: (B, V, H); e: (B, V, V, H);
        """
        batch_size, nodes_num, _, _ = e.shape
        graph = torch.ones(size=(batch_size, nodes_num, nodes_num)).to(e.device)
        # gnn layer
        for layer, out_layer in zip(self.layers, self.per_layer_out):
            layer: GNNDenseLayer
            e_in = e
            e = layer.asym_forward(e, edges_feature, graph)
            e = e_in + out_layer(e)
        
        # return
        return e
    

def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module