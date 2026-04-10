"""
ZINC-Full Benchmark: HCN (α1/α2) + HPM + Cycle-Size Virtual Edge Encoding
Methodology: FC-Cycles + Proven HCNLayer + HPM + Typed Virtual Edges
Optimized for AWS EC2 g5.12xlarge (4× A10G, PyTorch 2.1.0, PyG 2.7.0)

Uses the full ZINC dataset (~250K graphs, subset=False) instead of the 12k subset.
conv_up upgraded from GraphConv to GINEConv with virtual_edge_emb(cycle_size).
"""

import os
import random
import time
import queue
import numpy as np
import math
import platform
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import networkx as nx
import gc

from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphConv, GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC

# ==========================================
# 0. Memory Management Utilities
# ==========================================
def aggressive_gc():
    gc.collect()
    gc.collect()
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

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

# ==========================================
# 2. Cycle Extraction (FC-Cycles + Harmonic Basis + Cycle-Size Labels)
# ==========================================
def process_zinc_split(dataset_subset, split_name):
    total = len(dataset_subset)
    print(f"    Extracting FC-Cycles + Harmonic Basis + Cycle Sizes for {split_name} ({total} graphs)...")
    modified = []
    max_harmonics = 8

    for i, data in enumerate(dataset_subset):
        G = to_networkx(data, to_undirected=True)
        cycles = nx.minimum_cycle_basis(G)

        virtual_edges = []
        virtual_edge_sizes = []  # cycle size per virtual edge
        for cycle in cycles:
            csize = len(cycle)
            for i_idx in range(len(cycle)):
                for j_idx in range(i_idx + 1, len(cycle)):
                    virtual_edges.append((cycle[i_idx], cycle[j_idx]))
                    virtual_edges.append((cycle[j_idx], cycle[i_idx]))
                    virtual_edge_sizes.append(csize)
                    virtual_edge_sizes.append(csize)

        v_idx = (
            torch.tensor(virtual_edges, dtype=torch.long).t().contiguous()
            if virtual_edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        data.v_edge_index = v_idx

        # Store cycle-size labels for virtual edges
        data.v_edge_attr = (
            torch.tensor(virtual_edge_sizes, dtype=torch.long)
            if virtual_edge_sizes
            else torch.empty(0, dtype=torch.long)
        )

        # HPM: Compute harmonic basis of L_up
        num_nodes = data.num_nodes
        if virtual_edges and num_nodes > 1:
            A_up = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for (u, v) in virtual_edges:
                if u < num_nodes and v < num_nodes:
                    A_up[u, v] = 1.0
            A_up = np.maximum(A_up, A_up.T)
            D_up = np.diag(A_up.sum(axis=1))
            L_up = D_up - A_up
            eigenvalues, eigenvectors = np.linalg.eigh(L_up)
            k = min(max_harmonics, len(eigenvalues))
            harm_basis = eigenvectors[:, :k]
            if k < max_harmonics:
                padding = np.zeros((num_nodes, max_harmonics - k), dtype=np.float32)
                harm_basis = np.concatenate([harm_basis, padding], axis=1)
            data.harmonic_basis = torch.tensor(harm_basis, dtype=torch.float32)
        else:
            data.harmonic_basis = torch.zeros((num_nodes, max_harmonics), dtype=torch.float32)

        modified.append(data)
        if (i + 1) % 10000 == 0:
            print(f"      Processed {i + 1}/{total} graphs...")

    return modified

def prepare_zinc():
    cache_path = 'data/ZINC_FULL_FC_CYCLES_v4_SIZED.pt'

    if os.path.exists(cache_path):
        print("  [+] Loading cached ZINC-Full subsets (FC-Cycles + Sized) from disk...")
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    print("  [!] No cache found. Processing ZINC-Full...")
    train_dataset = ZINC(root='data/ZINC_FULL', subset=False, split='train')
    val_dataset = ZINC(root='data/ZINC_FULL', subset=False, split='val')
    test_dataset = ZINC(root='data/ZINC_FULL', subset=False, split='test')

    train_list = process_zinc_split(train_dataset, "train")
    val_list = process_zinc_split(val_dataset, "val")
    test_list = process_zinc_split(test_dataset, "test")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {'train': train_list, 'val': val_list, 'test': test_list}
    torch.save(cache_data, cache_path)
    print(f"  [+] Saved cache to {cache_path}")
    return cache_data

# ==========================================
# 3. Network Architecture
# ==========================================
class BaselineGINELayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(nn1, train_eps=True)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr_emb):
        h = self.conv(x, edge_index, edge_attr_emb)
        return F.relu(self.bn(x + h))

class HCNLayerSized(nn.Module):
    """HCN layer with cycle-size-aware virtual edge convolution.
    conv_down: GINEConv on physical edges (with bond features)
    conv_up:   GINEConv on virtual edges (with cycle-size features)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        nn_down = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv_down = GINEConv(nn_down, train_eps=True)

        # Upgraded: GINEConv for virtual edges (takes cycle-size embedding)
        nn_up = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv_up = GINEConv(nn_up, train_eps=True)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, edge_index, edge_attr_emb, v_idx, v_edge_emb):
        h_down = self.conv_down(x, edge_index, edge_attr_emb)
        if v_idx.size(1) > 0:
            h_up = self.conv_up(x, v_idx, v_edge_emb)
        else:
            h_up = torch.zeros_like(h_down)
        out = x + (self.alpha1 * h_down) + (self.alpha2 * h_up)
        return F.relu(self.bn(out))

class HarmonicProjectionModule(nn.Module):
    """HPM: Spectral tunneling via harmonic kernel of L_up."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, harmonic_basis, batch):
        device = x.device
        k = harmonic_basis.size(1)
        U = harmonic_basis
        H_expanded = U.unsqueeze(2) * x.unsqueeze(1)
        num_graphs = batch.max().item() + 1
        batch_expanded = batch.unsqueeze(1).unsqueeze(2).expand_as(H_expanded)
        H_spec = torch.zeros(num_graphs, k, x.size(1), device=device)
        H_spec.scatter_add_(0, batch_expanded, H_expanded)
        H_spec_flat = H_spec.view(num_graphs * k, -1)
        H_mixed_flat = self.mlp(H_spec_flat)
        H_mixed = H_mixed_flat.view(num_graphs, k, -1)
        H_per_node = H_mixed[batch]
        x_harm = (U.unsqueeze(2) * H_per_node).sum(dim=1)
        return x_harm

class ZINC_Model(nn.Module):
    """HCN (α1/α2) + HPM + Cycle-Size Virtual Edge Encoding."""
    def __init__(self, hidden_dim, num_layers=6, is_hcn=False):
        super().__init__()
        self.is_hcn = is_hcn
        self.node_emb = nn.Embedding(28, hidden_dim)
        self.edge_emb = nn.Embedding(4, hidden_dim)

        if self.is_hcn:
            # Cycle-size embedding: cycles from size 3 to 20
            self.virtual_edge_emb = nn.Embedding(21, hidden_dim)
            self.layers = nn.ModuleList([HCNLayerSized(hidden_dim) for _ in range(num_layers)])
            self.hpm = HarmonicProjectionModule(hidden_dim)
        else:
            self.layers = nn.ModuleList([BaselineGINELayer(hidden_dim) for _ in range(num_layers)])

        pool_dim = hidden_dim * 6 if is_hcn else hidden_dim * 3
        self.regressor = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = data.x.squeeze(-1)
        edge_attr = data.edge_attr.squeeze(-1)
        edge_index = data.edge_index
        batch = data.batch

        x = self.node_emb(x)
        edge_attr_emb = self.edge_emb(edge_attr)

        if self.is_hcn:
            # Embed virtual edge cycle sizes (clamp to prevent OOB for large macrocycles)
            if data.v_edge_attr.numel() > 0:
                safe_v_edge_attr = torch.clamp(data.v_edge_attr, max=20)
                v_edge_emb = self.virtual_edge_emb(safe_v_edge_attr)
            else:
                v_edge_emb = None
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr_emb, data.v_edge_index, v_edge_emb)
        else:
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr_emb)

        x_spatial = x

        if self.is_hcn:
            x_harm = self.hpm(x_spatial, data.harmonic_basis, batch)
            x_fused = torch.cat([x_spatial, x_harm], dim=1)
        else:
            x_fused = x_spatial

        x_pool = torch.cat([
            global_mean_pool(x_fused, batch),
            global_add_pool(x_fused, batch),
            global_max_pool(x_fused, batch),
        ], dim=1)

        out = self.regressor(x_pool).squeeze(-1)
        return out, x_spatial

# ==========================================
# 4. Training Loop
# ==========================================
def warmup_cosine_factor(epoch, warmup_epochs=20, total_epochs=400, start_factor=0.1, eta_min_factor=0.01):
    if epoch < warmup_epochs:
        return start_factor + (1.0 - start_factor) * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return eta_min_factor + 0.5 * (1.0 - eta_min_factor) * (1.0 + math.cos(math.pi * progress))

def evaluate_mae(model, loader, device):
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data)
            total_mae += F.l1_loss(out, data.y, reduction='sum').item()
    return total_mae / len(loader.dataset)

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs=400, log_prefix="", hodge_coef=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: warmup_cosine_factor(e, warmup_epochs=20, total_epochs=epochs))

    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 60
    print_interval = 20

    init_val_mae = evaluate_mae(model, val_loader, device)
    print(f"  {log_prefix} Epoch 000 | Train MAE: -.---- | Val MAE: {init_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} (init)")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out, node_embeddings = model(data)
            loss = F.l1_loss(out, data.y)

            if hasattr(model, 'is_hcn') and model.is_hcn and data.v_edge_index.size(1) > 0:
                row, col = data.v_edge_index
                h_norm = F.normalize(node_embeddings, p=2, dim=1)
                hodge_loss = torch.mean(torch.sum((h_norm[row] - h_norm[col]) ** 2, dim=1))
                loss += hodge_coef * hodge_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += F.l1_loss(out.detach(), data.y, reduction='sum').item()

        train_mae = total_loss / len(train_loader.dataset)
        val_mae = evaluate_mae(model, val_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % print_interval == 0:
            print(f"  {log_prefix} Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= early_stopping_patience:
            print(f"  {log_prefix} [!] Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    test_mae = evaluate_mae(model, test_loader, device)
    return test_mae, best_val_mae

# ==========================================
# 5. Per-Seed Worker
# ==========================================
def run_single_seed(seed, gpu_id, cache_path, num_layers, hidden_dim, hodge_coef, run_baseline, result_queue):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    tag = f"[GPU:{gpu_id} Seed:{seed} L:{num_layers} D:{hidden_dim} HC:{hodge_coef}]"
    print(f"\n{tag} Starting on {device}...")

    set_seed(seed)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)

    train_loader = DataLoader(cache['train'], batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, multiprocessing_context='fork')
    val_loader = DataLoader(cache['val'], batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, multiprocessing_context='fork')
    test_loader = DataLoader(cache['test'], batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, multiprocessing_context='fork')

    gcn_test_mae = None
    if run_baseline:
        print(f"  {tag} Training Baseline GINE...")
        gcn = ZINC_Model(hidden_dim, num_layers=num_layers, is_hcn=False).to(device)
        gcn_test_mae, _ = train_and_evaluate(gcn, train_loader, val_loader, test_loader, device, epochs=400, log_prefix=f"[GPU:{gpu_id} Seed:{seed} Baseline]", hodge_coef=0.0)
        print(f"  {tag} => Baseline Test MAE: {gcn_test_mae:.4f}")
        del gcn; torch.cuda.empty_cache(); gc.collect()

    set_seed(seed)
    print(f"  {tag} Training HCN+HPM+Sized (hodge_coef={hodge_coef})...")
    hcn = ZINC_Model(hidden_dim, num_layers=num_layers, is_hcn=True).to(device)
    hcn_test_mae, _ = train_and_evaluate(hcn, train_loader, val_loader, test_loader, device, epochs=400, log_prefix=tag, hodge_coef=hodge_coef)
    print(f"  {tag} => HCN+HPM+Sized Test MAE: {hcn_test_mae:.4f}")

    avg_a1 = np.mean([layer.alpha1.item() for layer in hcn.layers])
    avg_a2 = np.mean([layer.alpha2.item() for layer in hcn.layers])
    del hcn; torch.cuda.empty_cache(); gc.collect()

    result = {'seed': seed, 'gpu_id': gpu_id, 'L': num_layers, 'D': hidden_dim, 'hodge_coef': hodge_coef, 'hcn_test_mae': hcn_test_mae, 'alpha1': avg_a1, 'alpha2': avg_a2}
    if gcn_test_mae is not None:
        result['gcn_test_mae'] = gcn_test_mae
    result_queue.put(result)

    if gcn_test_mae is not None:
        print(f"\n{tag} ✓ COMPLETE — Baseline: {gcn_test_mae:.4f}, HCN+HPM+Sized: {hcn_test_mae:.4f}, Improvement: {gcn_test_mae - hcn_test_mae:+.4f}")
    else:
        print(f"\n{tag} ✓ COMPLETE — HCN+HPM+Sized: {hcn_test_mae:.4f}")

# ==========================================
# 6. Grand Execution Loop
# ==========================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    script_start_time = time.time()

    run_baseline = True  # Already have baseline from previous runs
    num_layers_list = [6]
    hidden_dims_list = [192]
    hodge_coefs = [0.01]
    seeds = [42]

    num_gpus = torch.cuda.device_count()
    print(f"\n{'=' * 85}")
    print(f" ZINC-Full (HCN + HPM + Cycle-Size Encoding) | LOWER IS BETTER")
    print(f" Depth (L): {num_layers_list}")
    print(f" Width (D): {hidden_dims_list}")
    print(f" Hodge coefficients: {hodge_coefs}")
    print(f" Run baseline: {run_baseline}")
    print(f" GPUs detected: {num_gpus}")
    print(f" Seeds: {seeds}")
    print(f"{'=' * 85}")

    cache = prepare_zinc()
    cache_path = 'data/ZINC_FULL_FC_CYCLES_v4_SIZED.pt'
    del cache; aggressive_gc()

    grid_results = {}
    baseline_results = {}

    for L in num_layers_list:
        for D in hidden_dims_list:
            for hc_idx, hc in enumerate(hodge_coefs):
                config_name = f"L={L}_D={D}_HC={hc}"
                print(f"\n{'#' * 85}")
                print(f" TESTING: L={L}, D={D}, HC={hc}")

                do_baseline = run_baseline and (hc_idx == 0)
                if do_baseline:
                    print(f" (Including Baseline GINE for L={L}, D={D})")
                print(f"{'#' * 85}")

                result_queue = mp.Queue()

                if num_gpus > 1:
                    batch_size_gpus = min(num_gpus, len(seeds))
                    seed_batches = [seeds[i:i + batch_size_gpus] for i in range(0, len(seeds), batch_size_gpus)]
                    all_batch_results = []

                    for batch_idx, seed_batch in enumerate(seed_batches):
                        print(f"\n--- GPU Batch {batch_idx + 1}/{len(seed_batches)}: seeds {seed_batch} ---")

                        processes = []
                        for i, seed in enumerate(seed_batch):
                            gpu_id = i % num_gpus
                            p = mp.Process(
                                target=run_single_seed,
                                args=(seed, gpu_id, cache_path, L, D, hc, do_baseline, result_queue),
                            )
                            p.start()
                            processes.append(p)

                        alive = set(range(len(processes)))
                        while alive:
                            try:
                                while True:
                                    all_batch_results.append(result_queue.get_nowait())
                            except queue.Empty:
                                pass
                            still_alive = set()
                            for idx in alive:
                                if processes[idx].is_alive():
                                    still_alive.add(idx)
                                else:
                                    processes[idx].join(timeout=0)
                            alive = still_alive
                            if alive:
                                time.sleep(1)

                        try:
                            while True:
                                all_batch_results.append(result_queue.get_nowait())
                        except queue.Empty:
                            pass

                        print(f"--- GPU Batch {batch_idx + 1} complete ---")

                    coef_results = list(all_batch_results)
                    try:
                        while True:
                            coef_results.append(result_queue.get_nowait())
                    except queue.Empty:
                        pass

                else:
                    print(f"\n  [!] Single GPU detected — running seeds sequentially")
                    for seed in seeds:
                        run_single_seed(seed, 0, cache_path, L, D, hc, do_baseline, result_queue)

                    coef_results = []
                    try:
                        while True:
                            coef_results.append(result_queue.get_nowait())
                    except queue.Empty:
                        pass

                coef_results.sort(key=lambda r: r['seed'])
                grid_results[config_name] = coef_results

                hcn_maes = [r['hcn_test_mae'] for r in coef_results]
                a1s = [r['alpha1'] for r in coef_results]
                a2s = [r['alpha2'] for r in coef_results]

                if do_baseline:
                    gcn_maes = [r['gcn_test_mae'] for r in coef_results if 'gcn_test_mae' in r]
                    if gcn_maes:
                        baseline_results[f"L={L}_D={D}"] = np.mean(gcn_maes)
                        print(f"\n  [L={L}, D={D}] Baseline GINE MAE: {np.mean(gcn_maes):.4f} ± {np.std(gcn_maes):.4f}")

                print(f"  [{config_name}] HCN+HPM+Sized MAE: {np.mean(hcn_maes):.4f} ± {np.std(hcn_maes):.4f} | α1={np.mean(a1s):.2f} | α2={np.mean(a2s):.2f}")

    # Final Table
    print(f"\n{'=' * 115}")
    print(f" ZINC-Full ABLATION: Original HCN + HPM (LOWER IS BETTER)")
    print(f"{'=' * 115}")
    print(f"| {'L':>5} | {'D':>5} | {'HC':>6} | {'Baseline MAE':>12} | {'HCN+HPM':>9} | {'Std':>8} | {'Improv':>8} | {'α1':>5} | {'α2':>5} |")
    print(f"|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*14}|{'-'*11}|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*7}|")

    best_config = None
    best_mae = float('inf')
    for L in num_layers_list:
        for D in hidden_dims_list:
            baseline_mae = baseline_results.get(f"L={L}_D={D}", None)
            baseline_str = f"{baseline_mae:.4f}" if baseline_mae is not None else "N/A"
            for hc in hodge_coefs:
                config_name = f"L={L}_D={D}_HC={hc}"
                results = grid_results.get(config_name, [])
                if not results: continue
                hcn_maes = [r['hcn_test_mae'] for r in results]
                a1s = [r['alpha1'] for r in results]
                a2s = [r['alpha2'] for r in results]
                mean_mae = np.mean(hcn_maes)
                std_mae = np.std(hcn_maes)
                margin_str = "N/A"
                if baseline_mae is not None:
                    margin_str = f"{baseline_mae - mean_mae:+.4f}"
                if mean_mae < best_mae:
                    best_mae = mean_mae
                    best_config = config_name
                print(f"| {L:>5} | {D:>5} | {hc:>6.4f} | {baseline_str:>12} | {mean_mae:>9.4f} | {std_mae:>8.4f} | {margin_str:>8} | {np.mean(a1s):>5.2f} | {np.mean(a2s):>5.2f} |")

    print(f"\n  Best Configuration: {best_config} → HCN+HPM MAE: {best_mae:.4f}")
    print(f"{'=' * 115}")
    elapsed = time.time() - script_start_time
    print(f"\n  Total execution time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
