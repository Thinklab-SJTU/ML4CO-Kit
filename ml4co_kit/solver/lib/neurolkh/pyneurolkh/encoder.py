import torch
import torch.nn.functional as F
from typing import Union
from torch import nn, Tensor


class BatchNormNode(nn.Module):
    def __init__(self, hidden_dim: int):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x: Tensor) -> Tensor:
        x_trans = x.transpose(1, 2).contiguous()
        x_trans_bn = self.batch_norm(x_trans)
        x_trans_bn: Tensor
        x_bn = x_trans_bn.transpose(1, 2).contiguous()
        return x_bn


class NodeFeatures(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        aggregation: str = "mean", 
        sparse_factor: int = 20, 
        is_pdp: bool = False
    ):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.node_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.to_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.edge_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.is_pdp = is_pdp
        if self.is_pdp:
            self.pickup_embedding = nn.Linear(hidden_dim, hidden_dim, True)
            self.deliver_embedding = nn.Linear(hidden_dim, hidden_dim, True)
        self.sparse_factor = sparse_factor
        
    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor) -> Tensor:
        batch_size, nodes_num, hidden_dim = x.size()
        Ux = self.node_embedding(x)  # (B, V, H)
        Vx = self.to_embedding(x)  # (B, V, H)
        
        if self.is_pdp:
            Px = self.pickup_embedding(x[:, 1:nodes_num // 2 + 1, :])
            Dx = self.deliver_embedding(x[:, nodes_num // 2 + 1:, :])
        
        Ve = self.edge_embedding(e) # (B, V*K, H)
        Ve: Tensor
        Ve = F.softmax(Ve.view(batch_size, nodes_num, self.sparse_factor, hidden_dim), dim=2)
        Ve = Ve.view(batch_size, nodes_num * self.sparse_factor, hidden_dim)
        Vx = Vx[torch.arange(batch_size).view(-1, 1), edge_index] # (B, V*K, H)
        
        to = Ve * Vx
        to: Tensor
        to = to.view(batch_size, nodes_num, self.sparse_factor, hidden_dim).sum(2) # (B, N, H)
        x_new = Ux + to
        
        if self.is_pdp:
            x_new[:, 1:nodes_num // 2 + 1, :] += Dx
            x_new[:, nodes_num // 2 + 1:, :] += Px
        
        return x_new


class EdgeFeatures(nn.Module):
    def __init__(self, hidden_dim: int, sparse_factor: int = 20):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_from = nn.Linear(hidden_dim, hidden_dim, True)
        self.V_to = nn.Linear(hidden_dim, hidden_dim, True)
        self.inverse_U = nn.Linear(hidden_dim, hidden_dim, True)
        self.W_placeholder = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_placeholder.data.uniform_(-1, 1)
        self.sparse_factor = sparse_factor

    def forward(
        self, x: Tensor,e: Tensor, edge_index: Tensor, inverse_edge_index: Tensor
    ) -> Tensor:
        batch_size, _, hidden_dim = x.size()
        Ue = self.U(e) # (B, V*K, H)
        inverse_Ue = self.inverse_U(e) # (B, V*K, H)
        inverse_Ue = torch.cat(
            (inverse_Ue, self.W_placeholder.view(1, 1, hidden_dim).repeat(batch_size, 1, 1)), 1
        )  # (B, V*K+1, H)
        inverse_node_embedding = inverse_Ue[torch.arange(batch_size).view(batch_size, 1), inverse_edge_index]

        Vx_from = self.V_from(x) # (B, V, H)
        Vx_to = self.V_to(x) # (B, V, H)
        Vx = Vx_to[torch.arange(batch_size).view(-1, 1), edge_index] # (B, V*K, H)
        Vx: Tensor
        Vx_from: Tensor
        Vx = Vx.view(batch_size, -1, self.sparse_factor, 128) + Vx_from.view(batch_size, -1, 1, 128)
        Vx = Vx.view(batch_size, -1, 128)
        e_new = Ue + Vx + inverse_node_embedding
        return e_new


class SparseGCNLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 128, 
        aggregation: str = "mean", 
        sparse_factor: int = 20, 
        is_pdp: bool = False
    ):
        super(SparseGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation, sparse_factor, is_pdp)
        self.edge_feat = EdgeFeatures(hidden_dim, sparse_factor)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormNode(hidden_dim)

    def forward(
        self, x: Tensor, e: Tensor, edge_index: Tensor, inverse_edge_index: Tensor
    ) -> Tensor:
        e_in = e # (B, V*K, H)
        x_in = x # (B, V, H)

        x_tmp = self.node_feat(x_in, e_in, edge_index.long())
        x_tmp = self.bn_node(x_tmp)
        x = F.relu(x_tmp)
        x_new = x_in + x

        e_tmp = self.edge_feat(x_new, e_in, edge_index.long(), inverse_edge_index.long()) # (B, V*K, H)
        e_tmp = self.bn_edge(e_tmp)
        e = F.relu(e_tmp)
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, layer_num: int = 2):
        super(MLP, self).__init__()
        self.layer_num = layer_num
        U = []
        for _ in range(self.layer_num - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x: Tensor) -> Tensor:
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)
            Ux = F.relu(Ux)
        y = self.V(Ux)
        return y


class SparseGCNEncoder(nn.Module):
    def __init__(
        self,
        num_gcn_layers: int = 30,
        num_mlp_layers: int = 3,
        node_feature_dim: int = 2,
        edge_feature_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        aggregation: str = "mean", 
        sparse_factor: int = 20,
        is_pdp: bool = False,
    ):
        # super
        super(SparseGCNEncoder, self).__init__()

        # params
        self.num_gcn_layers = num_gcn_layers
        self.num_mlp_layers = num_mlp_layers
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.sparse_factor = sparse_factor

        # embedding layer
        self.nodes_embedding = nn.Linear(self.node_feature_dim, self.hidden_dim, bias=False)
        self.edges_embedding = nn.Linear(self.edge_feature_dim, self.hidden_dim, bias=False)
        
        # gcn layer
        gcn_layers = []
        for _ in range(self.num_gcn_layers):
            gcn_layers.append(
                SparseGCNLayer(self.hidden_dim, self.aggregation, sparse_factor, is_pdp)
            )
        self.gcn_layers = nn.ModuleList(gcn_layers)
        
        # output layer
        self.mlp_edges = MLP(self.hidden_dim, self.output_dim, self.num_mlp_layers)
        self.mlp_nodes = MLP(self.hidden_dim, self.output_dim, self.num_mlp_layers)

    def forward(
        self, x: Tensor, graph: Tensor, edge_index: Tensor, inverse_edge_index: Tensor
    ) -> Union[Tensor, Tensor]:
        """
        Args:
            x (Tensor): Input node coordinates (B x V x 2)
            graph (Tensor): Graph sparse adjacency matrices (B x (V*K))
            edge_index (Tensor): Edge indices (B x (V*K))
            inverse_edge_index (Tensor): Inverse Edge indices (B x (V*K))
        """
        # info
        batch_size, nodes_num, _ = x.size()

        # embedding layer
        x = self.nodes_embedding(x)  # (B, V, H)
        e = self.edges_embedding(graph.unsqueeze(dim=-1))  # (B, V*K, H)

        # gcn layer
        for idx in range(self.num_gcn_layers):
            x, e = self.gcn_layers[idx](x, e, edge_index, inverse_edge_index)
            
        # output layer
        y_pred_edges = self.mlp_edges.forward(e).view(batch_size, nodes_num, self.sparse_factor)
        y_pred_edges = torch.exp(y_pred_edges)
        y_pred_edges = y_pred_edges / (y_pred_edges.sum(2).view(batch_size, nodes_num, 1) + 1e-5)
        
        y_pred_edges = y_pred_edges.view(batch_size, nodes_num * self.sparse_factor, 1)
        y_pred_edges = torch.cat([1 - y_pred_edges, y_pred_edges], dim = 2)
        y_pred_edges = torch.log(y_pred_edges)

        y_pred_nodes = self.mlp_nodes(x)
        y_pred_nodes = 10 * torch.tanh(y_pred_nodes)

        return y_pred_nodes, y_pred_edges 