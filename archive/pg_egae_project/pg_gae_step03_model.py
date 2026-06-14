"""
PG-EGAE Step 3: Edge-Level Graph Autoencoder Architecture
===================================================
This module defines the strictly Edge-Level GAE to prevent contextual leakage.
- Encoder: GATv2Conv (incorporates both Node features and Edge features).
- Decoder: Pairwise MLP that predicts edge features from node embeddings.
- Loss: Computed strictly on Edge predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class EdgeGAE(nn.Module):
    def __init__(self, node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2, dropout=0.1):
        super(EdgeGAE, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Encoder (GATv2)
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(
            GATv2Conv(node_in_dim, hidden_dim, heads=4, edge_dim=edge_in_dim, concat=False)
        )
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=4, edge_dim=edge_in_dim, concat=False)
            )
            
        # Edge Decoder (MLP)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, edge_in_dim)
        )

    def encode(self, x, edge_index, edge_attr):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr=edge_attr)
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h # Node embeddings Z: (N, hidden_dim)

    def decode(self, z, edge_index):
        # z: (N, hidden_dim)
        # edge_index: (2, E)
        src, dst = edge_index[0], edge_index[1]
        
        # Concat source and destination embeddings
        z_pair = torch.cat([z[src], z[dst]], dim=-1) # (E, hidden_dim * 2)
        
        # Predict edge features
        pred_edge_attr = self.edge_decoder(z_pair) # (E, edge_in_dim)
        return pred_edge_attr

    def forward(self, data):
        z = self.encode(data.x, data.edge_index, data.edge_attr)
        pred_edge_attr = self.decode(z, data.edge_index)
        return pred_edge_attr

class EdgeReconstructionLoss(nn.Module):
    def __init__(self):
        super(EdgeReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none') # Compute loss per edge

    def forward(self, pred_edge_attr, true_edge_attr):
        # Loss per edge is the mean of squared errors across feature dimensions
        # pred_edge_attr: (E, 3), true_edge_attr: (E, 3)
        loss_per_edge = self.mse(pred_edge_attr, true_edge_attr).mean(dim=1) # (E,)
        # Return total loss for backprop, and per-edge loss for inference/thresholding
        return loss_per_edge.mean(), loss_per_edge

if __name__ == "__main__":
    # Quick sanity check
    from torch_geometric.data import Data
    model = EdgeGAE()
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    
    # Mock data
    x = torch.tensor([[1.0], [0.0], [0.0]])
    edge_index = torch.tensor([[1, 2], [0, 0]])
    edge_attr = torch.tensor([[5.0, 0.1, 1.0], [10.0, 0.5, 1.0]])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    pred = model(data)
    print("Output shape:", pred.shape) # Expected (2, 3)
