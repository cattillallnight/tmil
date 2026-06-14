"""
PG-EGAE Step 4: Unsupervised Training Pipeline
===================================================
Goal: Train K independent Edge-Level Graph Autoencoders.
Method:
- Load the normal Star-Graphs.
- Split them by Peer Group (Cluster 0, 1, 2, 3).
- Train an EdgeGAE model exclusively on graphs from that peer group.
- Save the K trained models.
"""

import sys
import os
import json
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pg_gae_step03_model import EdgeGAE, EdgeReconstructionLoss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r'c:\Users\Thuy Quyen\Downloads\completeproduce\tmil_eth')
RESULTS_DIR = BASE_DIR / "results"
CLUSTERS_FILE = RESULTS_DIR / "pg_gae_step01_clusters.json"
GRAPHS_FILE = RESULTS_DIR / "pg_gae_step02_normal_graphs.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

def load_data_by_cluster():
    print("Loading clusters and graphs...")
    with open(CLUSTERS_FILE, "r") as f:
        clusters = json.load(f)
        
    normal_graphs_dict = torch.load(GRAPHS_FILE, map_location='cpu', weights_only=False)
    
    cluster_graphs = {0: [], 1: [], 2: [], 3: []}
    
    for addr, graphs in normal_graphs_dict.items():
        if addr in clusters["normal_groups"]:
            c_id = clusters["normal_groups"][addr]
            cluster_graphs[c_id].extend(graphs)
            
    for c_id, g_list in cluster_graphs.items():
        print(f"Cluster {c_id}: {len(g_list)} graphs")
        
    return cluster_graphs

def train_cluster_model(c_id, graphs):
    print(f"\n========== Training Model for Cluster {c_id} ==========")
    if not graphs:
        print("No graphs. Skipping.")
        return
        
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    
    model = EdgeGAE(node_in_dim=1, edge_in_dim=3, hidden_dim=64, num_layers=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = EdgeReconstructionLoss().to(DEVICE)
    
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total_edges = 0
        
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            pred_edge_attr = model(batch)
            
            # Loss computation
            loss, _ = criterion(pred_edge_attr, batch.edge_attr)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.edge_index.size(1)
            total_edges += batch.edge_index.size(1)
            
        avg_loss = total_loss / max(1, total_edges)
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Edge MSE Loss: {avg_loss:.4f}")
        
    # Save model
    out_path = RESULTS_DIR / "checkpoints" / f"pg_gae_cluster_{c_id}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path.name}")

def main():
    cluster_graphs = load_data_by_cluster()
    
    for c_id in range(4):
        train_cluster_model(c_id, cluster_graphs[c_id])
        
    print("\n[OK] Training Complete for all Peer Groups.")

if __name__ == "__main__":
    main()
