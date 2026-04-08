import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphConv, global_add_pool
from torch_geometric.loader import DataLoader
import random
import numpy as np

# ==========================================
# 1. Setup & Device
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. Dynamic Dataset Loading
# ==========================================
def prepare_dataset(dataset_name):
    """Downloads, processes cycles, and caches dynamically based on name."""
    cache_path = f'data/{dataset_name}_cycles_v2.pt'
    
    if os.path.exists(cache_path):
        print(f"  [+] Loading cached {dataset_name}...")
        return torch.load(cache_path, map_location='cpu')

    print(f"  [!] No cache found. Processing {dataset_name} (Extracting Cycles)...")
    # use_node_attr ensures we get node features for datasets that have them
    dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)
    
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    
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
        
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # We save a dictionary now so we can remember the dynamic feature sizes
    cache_data = {
        'data_list': modified_dataset,
        'num_features': num_features,
        'num_classes': num_classes
    }
    torch.save(cache_data, cache_path)
    return cache_data

# ==========================================
# 3. Dynamic Architectures
# ==========================================
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        return self.classifier(global_add_pool(x, batch)), x

class StabilizedHCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.conv_d1 = GraphConv(in_channels, hidden_dim)
        self.conv_d2 = GraphConv(hidden_dim, hidden_dim)
        self.conv_u1 = GraphConv(in_channels, hidden_dim)
        self.conv_u2 = GraphConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.3)
        
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index, v_idx, batch = data.x, data.edge_index, data.v_edge_index, data.batch
        
        x_p1 = self.conv_d1(x, edge_index)
        x_c1 = self.conv_u1(x, v_idx) if v_idx.size(1) > 0 else torch.zeros_like(x_p1)
        h1 = F.relu(self.bn1(x_p1 + (self.alpha1 * x_c1)))
        h1 = self.dropout(h1)
        
        x_p2 = self.conv_d2(h1, edge_index)
        x_c2 = self.conv_u2(h1, v_idx) if v_idx.size(1) > 0 else torch.zeros_like(x_p2)
        h_nodes = F.relu(self.bn2(x_p2 + (self.alpha2 * x_c2)))
        
        return self.classifier(global_add_pool(h_nodes, batch)), h_nodes

# ==========================================
# 4. Training Loop (Quiet Mode for Multi-Seed)
# ==========================================
def train_and_evaluate(model, train_loader, test_loader, is_hcn=False, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    print_interval = epochs // 5 # Print at 20%, 40%,..,100% to keep console clean
    
    for epoch in range(1, epochs + 1):
        model.train()
        correct_train, total_train = 0, 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out, h_nodes = model(data)
            loss = F.cross_entropy(out, data.y)
            
            if is_hcn and data.v_edge_index.size(1) > 0:
                row, col = data.v_edge_index
                h_norm = F.normalize(h_nodes, p=2, dim=1)
                hodge_loss = torch.mean(torch.sum((h_norm[row] - h_norm[col])**2, dim=1))
                loss += 0.01 * hodge_loss 
                
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct_train += int((pred == data.y).sum())
            total_train += data.y.size(0)
            
        if epoch % print_interval == 0:
            train_acc = correct_train / total_train
            print(f"      Epoch {epoch:03d}/{epochs} | Train Acc: {train_acc:.4f}")

    # Test Evaluation
    model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, _ = model(data)
            pred = out.argmax(dim=1)
            correct_test += int((pred == data.y).sum())
            total_test += data.y.size(0)
            
    return correct_test / total_test

# ==========================================
# 5. Grand Execution Loop
# ==========================================
if __name__ == '__main__':
    # Target benchmarks
    datasets_to_test = ['ENZYMES'] # ['MUTAG', 'PROTEINS', 'ENZYMES']
    # seeds = [42, 43, 123, 124, 456, 457, 789,790, 999, 1000]
    seeds = [42, 123, 456, 789, 999]
    hidden_dims = [32, 64, 128]

    print("\n" + "="*70)
    print(" STARTING GRAND MULTI-BENCHMARK ABLATION")
    print("="*70)

    for dataset_name in datasets_to_test:
        print(f"\n\n>>> PREPARING DATASET: {dataset_name} <<<")
        cache = prepare_dataset(dataset_name)
        master_data_list = cache['data_list']
        in_channels = cache['num_features']
        out_channels = cache['num_classes']
        
        print(f"  Features: {in_channels} | Classes: {out_channels} | Graphs: {len(master_data_list)}")
        
        results_agg = {dim: {'gcn': [], 'hcn': [], 'margin': [], 'a1': [], 'a2': []} for dim in hidden_dims}
        
        for seed in seeds:
            print("Seed:", seed)
            set_seed(seed)
            dataset = master_data_list.copy()
            random.shuffle(dataset)
            split = int(len(dataset) * 0.8)
            
            train_loader = DataLoader(dataset[:split], batch_size=32, shuffle=True)
            test_loader = DataLoader(dataset[split:], batch_size=32, shuffle=False)
            
            for dim in hidden_dims:
                print(f"  --- Testing Dimension: {dim} ---")
                print("    -> Training Baseline GCN...")
                gcn = BaselineGCN(in_channels, dim, out_channels).to(device)
                gcn_acc = train_and_evaluate(gcn, train_loader, test_loader, is_hcn=False)
                
                print("    -> Training Stabilized HCN...")
                hcn = StabilizedHCN(in_channels, dim, out_channels).to(device)
                hcn_acc = train_and_evaluate(hcn, train_loader, test_loader, is_hcn=True)
                
                results_agg[dim]['gcn'].append(gcn_acc * 100)
                results_agg[dim]['hcn'].append(hcn_acc * 100)
                results_agg[dim]['margin'].append((hcn_acc - gcn_acc) * 100)
                results_agg[dim]['a1'].append(hcn.alpha1.item())
                results_agg[dim]['a2'].append(hcn.alpha2.item())

        # Print the final formatted table for THIS dataset
        print("\n" + "=" * 95)
        print(f" FINAL RESULTS: {dataset_name} (Mean ± Std over 5 seeds)")
        print("=" * 95)
        print(f"| Hidden Dim | Baseline GCN (Acc) | Stabilized HCN (Acc) | Margin | Alpha 1 | Alpha 2 |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for dim in hidden_dims:
            gcn_m, gcn_s = np.mean(results_agg[dim]['gcn']), np.std(results_agg[dim]['gcn'])
            hcn_m, hcn_s = np.mean(results_agg[dim]['hcn']), np.std(results_agg[dim]['hcn'])
            marg_m = np.mean(results_agg[dim]['margin'])
            a1_m, a1_s = np.mean(results_agg[dim]['a1']), np.std(results_agg[dim]['a1'])
            a2_m, a2_s = np.mean(results_agg[dim]['a2']), np.std(results_agg[dim]['a2'])
            
            gcn_str = f"{gcn_m:.2f}% ± {gcn_s:.2f}%"
            hcn_str = f"{hcn_m:.2f}% ± {hcn_s:.2f}%"
            marg_str = f"{marg_m:+.2f}%"
            a1_str = f"{a1_m:.2f}±{a1_s:.2f}"
            a2_str = f"{a2_m:.2f}±{a2_s:.2f}"
            
            print(f"| {dim:<10} | {gcn_str:>18} | {hcn_str:>20} | {marg_str:>6} | {a1_str:>10} | {a2_str:>10} |")
        print("=" * 95)