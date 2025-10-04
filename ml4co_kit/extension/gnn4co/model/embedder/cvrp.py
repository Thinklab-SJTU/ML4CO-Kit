from typing import Sequence
from torch import Tensor, nn
from ...model.embedder.base import GNN4COEmbedder
from ...model.embedder.utils import (
    PositionEmbeddingSine, ScalarEmbeddingSine1D
)


class CVRPEmbedder(GNN4COEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool):
        super(CVRPEmbedder, self).__init__(hidden_dim, sparse)
        
        # node embedder (position)
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2)

        # node embedder (demand)
        self.demand_embed = ScalarEmbeddingSine1D(hidden_dim)

        # node embedder (is depot)
        self.is_depot_embed = nn.Embedding(2, hidden_dim)

        # node embedder (merge)
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)

        # edge embedder
        self.distance_embed = nn.Sequential(
            ScalarEmbeddingSine1D(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.label_embed = nn.Sequential(
            ScalarEmbeddingSine1D(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: Tensor, e: Tensor, t: Tensor) -> Sequence[Tensor]:
        """
        Args:
            if time flag:
                x: (V, 4); e: (E, 2); t: (1,)
            else:
                x: (V, 4); e: (E,); t: None
        Return:
            if time flag:
                x: (V, H); e: (E, H); t: (H)
            else:
                x: (V, H); e: (E, H); t: None
        Note:
            4 dimensions: [x, y, demand, is depot] 
        """
        # node embedding (position)
        x_pos_embed: Tensor = self.pos_embed(x[:, :2].unsqueeze(0)) # (1, V, H)
        x_pos_embed = x_pos_embed.squeeze(0) # (V, H)

        # node embedding (demand)
        x_demand_embed = self.demand_embed(x[:, 2]) # (V, H)

        # node embedding (is_depot)
        x_is_depot_embed = self.is_depot_embed(x[:, 3].long()) # (V, H)

        # node embedding (merge)
        x_feature = x_pos_embed + x_demand_embed + x_is_depot_embed # (V, H)
        x = self.node_embed(x_feature) # (V, H)
        
        # edge embedding
        e1 = self.distance_embed(e[:, 0]) # (E, H) 
        e2 = self.label_embed(e[:, 1]) # (E, H) 
        e = self.edge_embed(e1 + e2) # (E, H) 

        # return
        return x, e