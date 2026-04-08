import os
import copy
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.datasets import ZINC
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphConv, GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

# ==========================================
# 1. Setup & Device Configuration
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. Dataset Processing (Cycle Extraction)
# ==========================================
def process_zinc_split(dataset):
    modified_dataset = []
    for data in dataset:
        G = to_networkx(data, to_undirected=True)
        cycles = nx.minimum_cycle_basis(G)
        
        virtual_edges = []
        for cycle in cycles:
            for i in range(len(cycle)):
                for j in range(i + 1, len(cycle)):
                    virtual_edges.append((cycle[i], cycle[j]))
                    virtual_edges.append((cycle[j], cycle[i]))
        
        v_idx = torch.tensor(virtual_edges, dtype=torch.long).t().contiguous() if virtual_edges else torch.empty((2, 0), dtype=torch.long)
        data.v_edge_index = v_idx
        modified_dataset.append(data)
    return modified_dataset

def prepare_zinc():
    cache_path = 'data/ZINC_cycles_v2.pt' 
    if os.path.exists(cache_path):
        print("  [+] Loading cached ZINC subsets (with edge features)...")
        return torch.load(cache_path, map_location='cpu')

    print("  [!] No cache found. Processing ZINC (Extracting Cycles)...")
    train_dataset = ZINC(root='data/ZINC', subset=True, split='train')
    val_dataset   = ZINC(root='data/ZINC', subset=True, split='val')
    test_dataset  = ZINC(root='data/ZINC', subset=True, split='test')
    
    train_list = process_zinc_split(train_dataset)
    val_list = process_zinc_split(val_dataset)
    test_list = process_zinc_split(test_dataset)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {'train': train_list, 'val': val_list, 'test': test_list}
    torch.save(cache_data, cache_path)
    return cache_data

# ==========================================
# 3. Network Architecture (Feature Dropout + BatchNorm)
# ==========================================
class BaselineGINELayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1), # Feature Regularization (Protects D=192)
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(nn1, train_eps=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr_emb):
        h = self.conv(x, edge_index, edge_attr_emb)
        return F.relu(self.bn(x + h))

class HCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1), # Feature Regularization (Protects D=192)
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv_down = GINEConv(nn1, train_eps=True)
        self.conv_up = GraphConv(hidden_dim, hidden_dim)
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        # Learnable gating mechanisms
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, edge_index, edge_attr_emb, v_idx):
        h_down = self.conv_down(x, edge_index, edge_attr_emb)
        h_up = self.conv_up(x, v_idx) if v_idx.size(1) > 0 else torch.zeros_like(h_down)
        
        out = x + (self.alpha1 * h_down) + (self.alpha2 * h_up)
        return F.relu(self.bn(out))

class ZINC_Model(nn.Module):
    def __init__(self, hidden_dim, num_layers=6, is_hcn=False):
        super().__init__()
        self.is_hcn = is_hcn
        self.node_emb = nn.Embedding(28, hidden_dim)
        self.edge_emb = nn.Embedding(4, hidden_dim)
        
        if self.is_hcn:
            self.layers = nn.ModuleList([HCNLayer(hidden_dim) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([BaselineGINELayer(hidden_dim) for _ in range(num_layers)])
            
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # Strong readout regularization
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.squeeze(-1), data.edge_index, data.edge_attr.squeeze(-1), data.batch
        
        x = self.node_emb(x)
        edge_attr_emb = self.edge_emb(edge_attr)
        
        for layer in self.layers:
            if self.is_hcn:
                x = layer(x, edge_index, edge_attr_emb, data.v_edge_index)
            else:
                x = layer(x, edge_index, edge_attr_emb)
                
        x_pool = torch.cat([
            global_mean_pool(x, batch), 
            global_add_pool(x, batch), 
            global_max_pool(x, batch)
        ], dim=1)
        
        out = self.regressor(x_pool)
        return out.squeeze(-1), x

# ==========================================
# 4. SOTA Training Loop (Pure Graphs + Fixed Anchor)
# ==========================================
def warmup_cosine_factor(epoch, warmup_epochs=20, total_epochs=400, start_factor=0.1, eta_min_factor=0.01):
    if epoch < warmup_epochs:
        return start_factor + (1.0 - start_factor) * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return eta_min_factor + 0.5 * (1.0 - eta_min_factor) * (1.0 + math.cos(math.pi * progress))

def evaluate_mae(model, loader):
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data)
            total_mae += F.l1_loss(out, data.y, reduction='sum').item()
    return total_mae / len(loader.dataset)

def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=400):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Warmup + Cosine Scheduler
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=lambda e: warmup_cosine_factor(e, warmup_epochs=20, total_epochs=epochs)
    )

    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 60 
    print_interval = 20
    
    # --- STRICT TOPOLOGICAL ANCHOR ---
    hodge_coef = 0.01 
    # ---------------------------------
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_mae_accum = 0.0 
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # NOTE: dropout_adj is REMOVED. We use 100% pure molecular graphs.
            
            out, h_nodes = model(data)
            
            # Safe Metric Computation (Detached from Autograd)
            with torch.no_grad():
                epoch_train_mae_accum += F.l1_loss(out.detach(), data.y, reduction='sum').item()

            # Original Loss Computation
            loss = F.l1_loss(out, data.y)
            
            # Fixed Hodge Regularization
            if hasattr(model, 'is_hcn') and model.is_hcn and data.v_edge_index.size(1) > 0:
                row, col = data.v_edge_index
                h_norm = F.normalize(h_nodes, p=2, dim=1)
                hodge_loss = torch.mean(torch.sum((h_norm[row] - h_norm[col])**2, dim=1))
                loss += hodge_coef * hodge_loss # Rigidly Fixed at 0.01
                
            loss.backward()
            optimizer.step()
            
        # Finalize Train MAE for the epoch
        train_mae = epoch_train_mae_accum / len(train_loader.dataset)
        
        val_mae = evaluate_mae(model, val_loader)
        scheduler.step() # Step every epoch for Cosine
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % print_interval == 0:
            print(f"      Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        if patience_counter >= early_stopping_patience:
            print(f"      [!] Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    test_mae = evaluate_mae(model, test_loader)
    return test_mae, best_val_mae

# ==========================================
# 5. Grand Execution Loop (Hero Run)
# ==========================================
if __name__ == '__main__':
    # Max-Potential Configuration
    # 32: The Capacity Floor (Expect bad MAE, high Alpha 2)
    # 64: The Efficiency Budget
    # 128: The Baseline Comparison
    # 192: The Max-Potential Target
    # 256: The Over-parameterization Ceiling
    # 512: The Capacity Ceiling (Expect overfitting, low Alpha 2)
    hidden_dims = [192] # [32, 64, 128, 192, 256, 512]
    num_layers = 6
    seeds = 42 # [42, 123, 456, 789, 999]

    print("\n" + "="*80)
    print(f" STARTING MAX-POTENTIAL ZINC BENCHMARK (L={num_layers}, D={hidden_dims[0]})")
    print("="*80)

    cache = prepare_zinc()
    train_loader = DataLoader(cache['train'], batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(cache['val'], batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(cache['test'], batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    results_agg = {dim: {'gcn': [], 'hcn': [], 'margin': [], 'a1': [], 'a2': []} for dim in hidden_dims}
    
    for seed in seeds:
        print(f"\n[Seed: {seed}]")
        set_seed(seed)
        
        for dim in hidden_dims:
            print(f"  --- Testing Unconstrained Dimension: {dim} ---")
            
            print("    -> Training Baseline GINE...")
            gcn = ZINC_Model(dim, num_layers=num_layers, is_hcn=False).to(device)
            gcn_test_mae, _ = train_and_evaluate(gcn, train_loader, val_loader, test_loader, epochs=400)
            print(f"       => Baseline Test MAE: {gcn_test_mae:.4f}")
            
            print("    -> Training Stabilized HCN...")
            hcn = ZINC_Model(dim, num_layers=num_layers, is_hcn=True).to(device)
            hcn_test_mae, _ = train_and_evaluate(hcn, train_loader, val_loader, test_loader, epochs=400)
            print(f"       => HCN Test MAE: {hcn_test_mae:.4f}")
            
            results_agg[dim]['gcn'].append(gcn_test_mae)
            results_agg[dim]['hcn'].append(hcn_test_mae)
            results_agg[dim]['margin'].append(gcn_test_mae - hcn_test_mae)
            
            avg_a1 = np.mean([layer.alpha1.item() for layer in hcn.layers])
            avg_a2 = np.mean([layer.alpha2.item() for layer in hcn.layers])
            results_agg[dim]['a1'].append(avg_a1)
            results_agg[dim]['a2'].append(avg_a2)

    print("\n" + "=" * 95)
    print(" FINAL RESULTS: MAX-POTENTIAL ZINC-12k (Mean ± Std over 5 seeds) | LOWER IS BETTER")
    print("=" * 95)
    print(f"| Hidden Dim | Baseline GINE (MAE)| Stabilized HCN (MAE) | Abs Improv. | Alpha 1 | Alpha 2 |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for dim in hidden_dims:
        gcn_m, gcn_s = np.mean(results_agg[dim]['gcn']), np.std(results_agg[dim]['gcn'])
        hcn_m, hcn_s = np.mean(results_agg[dim]['hcn']), np.std(results_agg[dim]['hcn'])
        marg_m = np.mean(results_agg[dim]['margin'])
        a1_m, a1_s = np.mean(results_agg[dim]['a1']), np.std(results_agg[dim]['a1'])
        a2_m, a2_s = np.mean(results_agg[dim]['a2']), np.std(results_agg[dim]['a2'])
        
        gcn_str = f"{gcn_m:.4f} ± {gcn_s:.4f}"
        hcn_str = f"{hcn_m:.4f} ± {hcn_s:.4f}"
        marg_str = f"{marg_m:+.4f}"
        a1_str = f"{a1_m:.2f}±{a1_s:.2f}"
        a2_str = f"{a2_m:.2f}±{a2_s:.2f}"
        
        print(f"| {dim:<10} | {gcn_str:>18} | {hcn_str:>20} | {marg_str:>11} | {a1_str:>10} | {a2_str:>10} |")
    print("=" * 95)