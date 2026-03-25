"""
PyTorch GAT Benchmark for comparison with ATHENA's GAT implementation.

This script creates the same synthetic graph and GAT architecture as the
ATHENA gat_node_classification example, trains it, and reports timing
and accuracy for direct comparison.

Requirements:
    pip install torch torch-geometric

Usage:
    python benchmarks/pytorch_gat_comparison.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────
# Hyperparameters (matching ATHENA example)
# ──────────────────────────────────────────────────────────────────────
NUM_NODES = 30
NUM_FEATURES_IN = 4
NUM_CLASSES = 3
NUM_HEADS = 2
HIDDEN_DIM = 8       # per head, total = HIDDEN_DIM * NUM_HEADS = 16
NODES_PER_CLASS = NUM_NODES // NUM_CLASSES
NUM_EPOCHS = 100
LEARNING_RATE = 5e-3
NEGATIVE_SLOPE = 0.2


# ──────────────────────────────────────────────────────────────────────
# Minimal GAT layer (no torch-geometric dependency)
# ──────────────────────────────────────────────────────────────────────
class GATLayer(nn.Module):
    """Single Graph Attention Network layer following Velickovic et al. (2018)."""

    def __init__(self, in_features, out_features, num_heads=1,
                 concat=True, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        if concat:
            assert out_features % num_heads == 0
            self.out_per_head = out_features // num_heads
        else:
            self.out_per_head = out_features

        # Learnable weight matrix W and attention vectors a_l, a_r per head
        self.W = nn.Parameter(
            torch.empty(num_heads, in_features, self.out_per_head)
        )
        self.a_l = nn.Parameter(torch.empty(num_heads, self.out_per_head))
        self.a_r = nn.Parameter(torch.empty(num_heads, self.out_per_head))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(-1))

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, F_in]
            edge_index: Edge indices [2, E]
        Returns:
            Node features [N, F_out]
        """
        N = x.size(0)
        src, dst = edge_index  # src -> dst

        # Project: [N, F_in] @ [K, F_in, F'] -> [K, N, F']
        Wh = torch.einsum("nf,kfh->knh", x, self.W)

        # Attention logits
        e_l = torch.einsum("knh,kh->kn", Wh, self.a_l)  # [K, N]
        e_r = torch.einsum("knh,kh->kn", Wh, self.a_r)  # [K, N]

        # e_ij = LeakyReLU(e_l[src] + e_r[dst])
        e = F.leaky_relu(
            e_l[:, src] + e_r[:, dst],
            negative_slope=self.negative_slope,
        )  # [K, E]

        # Softmax per destination node
        alpha = torch.zeros(self.num_heads, N, device=x.device)
        e_max = torch.full((self.num_heads, N), -1e9, device=x.device)
        e_max.scatter_reduce_(1, dst.unsqueeze(0).expand_as(e), e, reduce="amax")
        e_shifted = e - e_max[:, dst]
        exp_e = torch.exp(e_shifted)
        denom = torch.zeros(self.num_heads, N, device=x.device)
        denom.scatter_add_(1, dst.unsqueeze(0).expand_as(exp_e), exp_e)
        alpha = exp_e / (denom[:, dst] + 1e-16)  # [K, E]

        # Weighted aggregation
        weighted = alpha.unsqueeze(-1) * Wh[:, src, :]  # [K, E, F']
        out = torch.zeros(self.num_heads, N, self.out_per_head, device=x.device)
        out.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand_as(weighted),
            weighted,
        )  # [K, N, F']

        if self.concat:
            return out.permute(1, 0, 2).reshape(N, -1)  # [N, K*F']
        else:
            return out.mean(dim=0)  # [N, F']


class GATNet(nn.Module):
    """Two-layer GAT network matching the ATHENA example architecture."""

    def __init__(self):
        super().__init__()
        # Layer 1: multi-head, concat
        self.gat1 = GATLayer(
            NUM_FEATURES_IN,
            HIDDEN_DIM * NUM_HEADS,
            num_heads=NUM_HEADS,
            concat=True,
            negative_slope=NEGATIVE_SLOPE,
        )
        # Layer 2: single head, average (concat=False with 1 head is same)
        self.gat2 = GATLayer(
            HIDDEN_DIM * NUM_HEADS,
            NUM_CLASSES,
            num_heads=1,
            concat=False,
            negative_slope=NEGATIVE_SLOPE,
        )

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.softmax(self.gat2(x, edge_index), dim=-1)
        return x


# ──────────────────────────────────────────────────────────────────────
# Create synthetic graph (same as ATHENA example)
# ──────────────────────────────────────────────────────────────────────
def create_synthetic_graph():
    """Create the same community graph as the Fortran example."""
    rng = np.random.RandomState(SEED)

    # Node features
    features = np.full((NUM_NODES, NUM_FEATURES_IN), 0.1, dtype=np.float32)
    labels = np.zeros(NUM_NODES, dtype=np.int64)
    for v in range(NUM_NODES):
        class_id = v // NODES_PER_CLASS
        features[v, class_id] = 0.8
        features[v, NUM_FEATURES_IN - 1] = rng.rand() * 0.5
        labels[v] = class_id

    # Edges
    src_list, dst_list = [], []

    # Intra-community (60% connectivity)
    for c in range(NUM_CLASSES):
        start = c * NODES_PER_CLASS
        end = (c + 1) * NODES_PER_CLASS
        for i in range(start, end):
            for j in range(i + 1, end):
                if rng.rand() < 0.6:
                    src_list.extend([i, j])
                    dst_list.extend([j, i])

    # Inter-community (5% connectivity)
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            if i // NODES_PER_CLASS == j // NODES_PER_CLASS:
                continue
            if rng.rand() < 0.05:
                src_list.extend([i, j])
                dst_list.extend([j, i])

    # Self-loops
    for v in range(NUM_NODES):
        src_list.append(v)
        dst_list.append(v)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float32)

    # One-hot labels for MSE comparison
    labels_onehot = np.zeros((NUM_NODES, NUM_CLASSES), dtype=np.float32)
    for v in range(NUM_NODES):
        labels_onehot[v, labels[v]] = 1.0
    y = torch.tensor(labels_onehot, dtype=torch.float32)

    return x, edge_index, y, labels


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PyTorch GAT Benchmark")
    print("=" * 60)

    device = torch.device("cpu")   # match ATHENA (CPU)

    x, edge_index, y, labels = create_synthetic_graph()
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

    print(f"Graph: {NUM_NODES} nodes, {edge_index.size(1)} edges (with self-loops)")
    print(f"Architecture: {NUM_FEATURES_IN} -> {HIDDEN_DIM}x{NUM_HEADS} heads -> {NUM_CLASSES}")

    model = GATNet().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # ── Training ──────────────────────────────────────────────────────
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    t_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = loss_fn(out, y)
        loss.backward()

        # Gradient clipping (matching ATHENA clip_type(-1, 1))
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4d}/{NUM_EPOCHS}  loss: {loss.item():.5f}")

    t_train = time.perf_counter() - t_start
    print(f"\nTraining time: {t_train:.4f} s")

    # ── Evaluation ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        correct = (pred == labels).sum()
        final_loss = loss_fn(out, y).item()

    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"Classification accuracy: {correct}/{NUM_NODES} = {100*correct/NUM_NODES:.1f}%")

    print("\nNode classification results:")
    print("  Node | True | Pred | Correct")
    print("  -----+------+------+--------")
    for v in range(NUM_NODES):
        if v < 10 or v % 10 == 9:
            c = "T" if pred[v] == labels[v] else "F"
            print(f"  {v+1:4d} | {labels[v]+1:4d} | {pred[v]+1:4d} | {c}")

    # ── Timing summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(f"  Training ({NUM_EPOCHS} epochs): {t_train:.4f} s")
    print(f"  Per epoch:               {t_train/NUM_EPOCHS*1000:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
