import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GraphConv, global_add_pool

# Compatibility fix for DataLoader in PyG 2.0.x
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

# Setup device (use GPU if available, though CPU is fast enough for this)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Dataset Generation
# ==========================================
def create_synthetic_dataset(num_graphs=200, num_nodes=50):
    dataset = []
    for i in range(num_graphs):
        label = i % 2
        if label == 1:
            # Class 1: Cycle Graph (H1 > 0)
            G = nx.cycle_graph(num_nodes)
            # Sparse Cycle Lifting: Connect all nodes in the cycle to form A_up
            virtual_edges = list(nx.complete_graph(num_nodes).edges())
        else:
            # Class 0: Path Graph (Tree, H1 = 0)
            G = nx.path_graph(num_nodes)
            virtual_edges = [] # No virtual edges for acyclic graphs

        # Physical Edges (A_down)
        edge_list = list(G.edges())
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Virtual Edges (A_up)
        if len(virtual_edges) > 0:
            v_edge_index = torch.tensor(virtual_edges, dtype=torch.long).t().contiguous()
            v_edge_index = torch.cat([v_edge_index, v_edge_index[[1, 0]]], dim=1)
        else:
            v_edge_index = torch.empty((2, 0), dtype=torch.long)

        # Constant node features to force reliance on topology
        x = torch.ones((num_nodes, 1), dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, v_edge_index=v_edge_index, y=y)
        dataset.append(data)
    return dataset

# ==========================================
# 2. Model Definitions (FIXED)
# ==========================================
class StandardGCN(torch.nn.Module):
    def __init__(self):
        super(StandardGCN, self).__init__()
        # GraphConv doesn't apply the aggressive degree penalty of GCNConv
        self.conv1 = GraphConv(1, 16)
        self.conv2 = GraphConv(16, 16)
        self.classifier = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Add pool preserves the structural scale much better than mean pool
        x = global_add_pool(x, batch)
        return self.classifier(x)

class SimplifiedHCN(torch.nn.Module):
    def __init__(self):
        super(SimplifiedHCN, self).__init__()
        self.conv_down1 = GraphConv(1, 16)
        self.conv_up1 = GraphConv(1, 16)
        self.conv_down2 = GraphConv(16, 16)
        self.conv_up2 = GraphConv(16, 16)
        self.classifier = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index, v_edge_index, batch = data.x, data.edge_index, data.v_edge_index, data.batch
        
        # --- Layer 1 ---
        x_down = self.conv_down1(x, edge_index)
        if v_edge_index.size(1) > 0:
            x_up = self.conv_up1(x, v_edge_index)
        else:
            x_up = torch.zeros_like(x_down)
        x = F.relu(x_down + x_up)
        
        # --- Layer 2 ---
        x_down = self.conv_down2(x, edge_index)
        if v_edge_index.size(1) > 0:
            x_up = self.conv_up2(x, v_edge_index)
        else:
            x_up = torch.zeros_like(x_down)
        x = F.relu(x_down + x_up)
        
        x = global_add_pool(x, batch)
        return self.classifier(x)

# ==========================================
# 3. Training Loop
# ==========================================
def train_and_evaluate(model, loader, optimizer):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
    return correct / total

# ==========================================
# 4. Execution
# ==========================================
if __name__ == '__main__':
    print("Generating Dataset...")
    dataset = create_synthetic_dataset(num_graphs=200, num_nodes=50)
    
    # 80/20 Train-Test Split
    train_loader = DataLoader(dataset[:160], batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset[160:], batch_size=16, shuffle=False)

    print("\n--- Testing Standard GCN Baseline ---")
    gcn_model = StandardGCN().to(device)
    optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
    for epoch in range(1, 51):
        acc = train_and_evaluate(gcn_model, train_loader, optimizer_gcn)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | GCN Train Accuracy: {acc:.4f}")
            
    gcn_test_acc = train_and_evaluate(gcn_model, test_loader, optimizer_gcn)
    print(f">> GCN Final Test Accuracy: {gcn_test_acc:.4f}\n")

    print("--- Testing Simplified HCN ---")
    hcn_model = SimplifiedHCN().to(device)
    optimizer_hcn = torch.optim.Adam(hcn_model.parameters(), lr=0.01)
    for epoch in range(1, 51):
        acc = train_and_evaluate(hcn_model, train_loader, optimizer_hcn)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | HCN Train Accuracy: {acc:.4f}")
            
    hcn_test_acc = train_and_evaluate(hcn_model, test_loader, optimizer_hcn)
    print(f">> HCN Final Test Accuracy: {hcn_test_acc:.4f}")