"""
PyTorch Geometric GNN matching ATHENA Fortran Duvenaud MPNN.

Uses torch_geometric.nn.MessagePassing for the message passing framework.
Each DuvenaudLayer maps to ATHENA's duvenaud_propagate + duvenaud_update.

Architecture mapping (ATHENA -> PyG):
  duvenaud_propagate  ->  message() + aggregate()
  duvenaud_update     ->  update()  (degree-specific W_d @ (aggr / d))
  sigmoid activation  ->  applied inside update()
  readout             ->  DuvenaudMPNN.forward (R @ z^T -> softmax -> sum)
  full_layer          ->  nn.Linear (with bias)
  leaky_relu          ->  F.leaky_relu

Usage:
    python pytorch_network.py              # Cross-validate forward pass
    python pytorch_network.py --train      # Train and compare loss curves
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import sys
import glob
import hashlib
from collections import OrderedDict
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter as pyg_scatter


# ===========================================================================
# ATHENA text ONNX parser
# ===========================================================================

def parse_athena_onnx(filepath):
    """Parse an ATHENA text-format ONNX file."""
    initialisers = OrderedDict()
    nodes = []
    with open(filepath, 'r') as f:
        content = f.read()
    for match in re.finditer(r'initializer\s*\{(.*?)\}', content, re.DOTALL):
        block = match.group(1)
        name_match = re.search(r'name:\s*"([^"]*)"', block)
        name = name_match.group(1) if name_match else "unknown"
        dims = [int(d) for d in re.findall(r'dims:\s*(\d+)', block)]
        float_match = re.search(r'float_data:\s*\[(.*?)\]', block, re.DOTALL)
        if float_match:
            values = [float(t) for t in re.findall(
                r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?',
                float_match.group(1))]
            data = np.array(values, dtype=np.float32)
        else:
            data = np.array([], dtype=np.float32)
        initialisers[name] = {'dims': dims, 'data': data}
    for match in re.finditer(r'node\s*\{(.*?)\n\s*\}', content, re.DOTALL):
        block = match.group(1)
        node = {
            'name': (re.search(r'name:\s*"([^"]*)"', block) or
                     type('', (), {'group': lambda s, i: ''})()).group(1),
            'op_type': (re.search(r'op_type:\s*"([^"]*)"', block) or
                        type('', (), {'group': lambda s, i: ''})()).group(1),
            'inputs': re.findall(r'input:\s*"([^"]*)"', block),
            'outputs': re.findall(r'output:\s*"([^"]*)"', block),
        }
        attrs = {}
        for ab in re.findall(r'attribute\s*\{(.*?)\}', block, re.DOTALL):
            an = re.search(r'name:\s*"([^"]*)"', ab)
            if an:
                iv = re.findall(r'\bints?:\s*(\d+)', ab)
                if iv:
                    attrs[an.group(1)] = [int(v) for v in iv]
                else:
                    sv = re.search(r'\bs:\s*"([^"]*)"', ab)
                    if sv:
                        attrs[an.group(1)] = sv.group(1)
        node['attributes'] = attrs
        nodes.append(node)
    return initialisers, nodes


# ===========================================================================
# Data conversion: ATHENA CSR -> PyG format
# ===========================================================================

def csr_to_pyg(vertex_features, edge_features, adj_ia, adj_ja):
    """
    Convert ATHENA-format graph to PyG tensors.

    ATHENA layout:
        vertex_features: [nv_feat, num_vertices] (features-first, Fortran)
        edge_features:   [ne_feat, num_edges]
        adj_ia:          [num_vertices + 1] 1-indexed CSR row pointers
        adj_ja:          [2, num_csr] 1-indexed [neighbor_vertex; edge_feat_idx]

    PyG layout:
        x:           [num_nodes, nv_feat]
        edge_index:  [2, num_directed_edges] (source, target)
        edge_attr:   [num_directed_edges, ne_feat]
        node_degree: [num_nodes] integer

    Edge ordering matches CSR traversal for numerical consistency with Fortran.
    """
    num_vertices = vertex_features.shape[1]
    src, tgt, attrs = [], [], []
    for v in range(num_vertices):
        start = adj_ia[v].item() - 1
        end = adj_ia[v + 1].item() - 1
        for w in range(start, end):
            src.append(adj_ja[0, w].item() - 1)   # neighbor = source
            tgt.append(v)                           # current vertex = target
            attrs.append(edge_features[:, adj_ja[1, w].item() - 1])
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.stack(attrs, dim=0)          # [num_csr, ne_feat]
    x = vertex_features.t().contiguous()           # [num_nodes, nv_feat]
    node_degree = torch.zeros(num_vertices, dtype=torch.long)
    for v in range(num_vertices):
        node_degree[v] = adj_ia[v + 1].item() - adj_ia[v].item()
    return x, edge_index, edge_attr, node_degree


# ===========================================================================
# Duvenaud message passing layer (torch_geometric.nn.MessagePassing)
#
# Maps to ATHENA Fortran:
#   message()   -> duvenaud_propagate: concat [h_neighbor; e_edge] per edge
#   aggregate() -> duvenaud_propagate: sum over neighbors for each vertex
#   update()    -> duvenaud_update: W_d @ (aggregated / d), then sigmoid
# Fortran sources:
#   athena_diffstruc_extd_sub_duvenaud.f90 (propagate, update ops)
#   athena_duvenaud_msgpass_layer.f90      (layer orchestration)
# ===========================================================================

class DuvenaudLayer(MessagePassing):
    """
    Single Duvenaud message passing timestep via PyG MessagePassing.

    Per time step t, for each vertex v:
        z_v^(t) = sigmoid( W_d @ (1/d * sum_{w in N(v)} [z_w^(t-1); e_wv]) )

    where d = clamp(deg(v), min_degree, max_degree) - min_degree + 1.

    Weight layout: flat 1D, Fortran column-major order, all degree buckets
        concatenated: [W_1 | W_2 | ... | W_D], each W_d has shape (F_out, F_in).
    No bias. Activation: sigmoid (ATHENA default_message_actv_name).
    """

    def __init__(self, in_channels, edge_channels, max_degree, min_degree=1):
        super().__init__(aggr='add', flow='source_to_target')
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.num_degree_buckets = max_degree - min_degree + 1
        msg_size = (in_channels * (in_channels + edge_channels)
                    * self.num_degree_buckets)
        self.weight = nn.Parameter(torch.zeros(msg_size))

    def forward(self, x, edge_index, edge_attr, node_degree):
        """
        x:           [num_nodes, in_channels]
        edge_index:  [2, num_edges] source -> target
        edge_attr:   [num_edges, edge_channels]
        node_degree: [num_nodes] CSR degree per node (including self-loops)
        """
        self._node_degree = node_degree
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_j, edge_attr):
        """
        Concatenate source (neighbor) vertex features with edge features.
        Maps to ATHENA duvenaud_propagate: building [h_neighbor; e_edge].

        x_j:       [num_edges, in_channels]  (source node features)
        edge_attr: [num_edges, edge_channels]
        Returns:   [num_edges, in_channels + edge_channels]
        """
        return torch.cat([x_j, edge_attr], dim=-1)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Sum aggregation over incoming messages per target node.
        Maps to ATHENA duvenaud_propagate: sum over adj_ja entries per vertex.
        Uses PyG built-in scatter-add (aggr='add').
        """
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def update(self, aggr_out):
        """
        Degree-specific linear transform + sigmoid activation.
        Maps to ATHENA duvenaud_update followed by sigmoid:
            new_x[v] = W_d @ (aggregated[v] / d)
            z[v] = sigmoid(new_x[v])

        Weight extraction from flat column-major storage:
            interval = F_out * F_in
            W_d = weight[interval*(d-1) : interval*d].view(F_in, F_out).t()
        The .view(F_in, F_out).t() converts Fortran column-major to [F_out, F_in].

        aggr_out: [num_nodes, in_channels + edge_channels]
        Returns:  [num_nodes, in_channels]
        """
        nv = self.in_channels
        ne = self.edge_channels
        interval = nv * (nv + ne)
        degree = self._node_degree
        num_nodes = aggr_out.shape[0]

        result = torch.zeros(num_nodes, nv, dtype=aggr_out.dtype)
        for v in range(num_nodes):
            d = (max(self.min_degree, min(degree[v].item(), self.max_degree))
                 - self.min_degree + 1)
            w_slice = self.weight[interval * (d - 1): interval * d]
            # Fortran column-major -> PyTorch: view(F_in, F_out).t() = [F_out, F_in]
            W = w_slice.view(nv + ne, nv).t()
            result[v] = W @ (aggr_out[v] / float(d))

        return torch.sigmoid(result)


# ===========================================================================
# Full Duvenaud MPNN (multiple timesteps + readout)
#
# Maps to ATHENA: duvenaud_msgpass_layer_type (node_1 in ONNX)
# ONNX parameter mapping:
#   layers[t].weight     -> node_1_param{t+1}       (message weights)
#   readout_weights[t]   -> node_1_param{t+1+T}     (readout weights)
# ===========================================================================

class DuvenaudMPNN(nn.Module):
    """
    Duvenaud MPNN: T message passing steps + graph readout.

    Message passing (per timestep):
        ATHENA: update_message_duvenaud -> propagate -> update -> activation
        PyG:    DuvenaudLayer.forward (message + aggregate + update w/ sigmoid)

    Readout (ATHENA: update_readout_duvenaud):
        For each timestep t:
            R_t @ z_t^T -> softmax(dim=0, over output features) -> sum over nodes
        Sum across all timesteps.
    """

    def __init__(self, in_channels, edge_channels, max_degree,
                 min_degree=1, num_timesteps=4, num_outputs=10):
        super().__init__()
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs

        self.layers = nn.ModuleList([
            DuvenaudLayer(in_channels, edge_channels, max_degree, min_degree)
            for _ in range(num_timesteps)
        ])
        # Readout weights: [num_outputs, in_channels] per timestep, stored flat
        ro_size = num_outputs * in_channels
        self.readout_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(ro_size)) for _ in range(num_timesteps)
        ])

    def forward(self, x, edge_index, edge_attr, node_degree, batch=None):
        """
        x:           [num_nodes, in_channels]
        edge_index:  [2, num_edges]
        edge_attr:   [num_edges, edge_channels]
        node_degree: [num_nodes]
        batch:       [num_nodes] graph index per node (None = single graph)
        Returns:     [batch_size, num_outputs] graph-level predictions
        """
        nv = self.in_channels
        num_nodes = x.shape[0]

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        batch_size = int(batch.max().item()) + 1

        z_list = []
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, node_degree)
            z_list.append(x)

        # Readout: maps to ATHENA update_readout_duvenaud
        # matmul(readout_W, z) -> activation_readout(softmax) -> per-graph sum
        output = torch.zeros(batch_size, self.num_outputs, dtype=x.dtype, device=x.device)
        for t in range(self.num_timesteps):
            z = z_list[t]                                     # [num_nodes, nv]
            r_flat = self.readout_weights[t]
            # Fortran column-major: view(nv, num_outputs).t() = [num_outputs, nv]
            R = r_flat.view(nv, self.num_outputs).t()
            out_t = R @ z.t()                                 # [num_outputs, num_nodes]
            # ATHENA softmax: normalize each node's output features (dim=0)
            out_t = F.softmax(out_t, dim=0)
            out_t = out_t.t()                                 # [num_nodes, num_outputs]
            # Scatter-sum over nodes, grouped by graph
            out_t = pyg_scatter(out_t, batch, dim=0,
                                dim_size=batch_size, reduce='sum')  # [batch_size, num_outputs]
            output = output + out_t

        return output


# ===========================================================================
# Full network: MPNN + FC layers
#
# Maps to ATHENA network architecture:
#   node_1: DuvenaudMPNN (4 timesteps, message passing + readout)
#   node_2: Linear(10, 128, bias=True) + LeakyReLU  [full_layer + activation]
#   node_3: Linear(128, 64, bias=True) + LeakyReLU  [full_layer + activation]
#   node_4: Linear(64,  1,  bias=True) + LeakyReLU  [full_layer + activation]
# ===========================================================================

class Net(nn.Module):
    """Full GNN matching ATHENA msgpass_chemical example."""

    def __init__(self, v_in=6, e_in=1):
        super().__init__()
        self.mpnn = DuvenaudMPNN(v_in, e_in, max_degree=10, min_degree=1,
                                 num_timesteps=4, num_outputs=10)
        self.fc1 = nn.Linear(10, 128)    # ATHENA node_2
        self.fc2 = nn.Linear(128, 64)    # ATHENA node_3
        self.fc3 = nn.Linear(64, 1)      # ATHENA node_4

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_degree = data.node_degree
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None
        out = self.mpnn(x, edge_index, edge_attr, node_degree, batch)  # [batch_size, 10]
        out = F.leaky_relu(self.fc1(out))             # [batch_size, 128]
        out = F.leaky_relu(self.fc2(out))             # [batch_size, 64]
        out = F.leaky_relu(self.fc3(out))             # [batch_size, 1]
        return out.squeeze(-1)                        # [batch_size]

    def load_from_athena_onnx(self, initialisers):
        """Load all parameters from parsed ATHENA ONNX initialisers."""
        nts = self.mpnn.num_timesteps
        with torch.no_grad():
            for t in range(nts):
                key = f"node_1_param{t + 1}"
                self.mpnn.layers[t].weight.copy_(
                    torch.from_numpy(initialisers[key]['data']))
                key = f"node_1_param{t + 1 + nts}"
                self.mpnn.readout_weights[t].copy_(
                    torch.from_numpy(initialisers[key]['data']))

            for fc, prefix in [(self.fc1, "node_2"),
                               (self.fc2, "node_3"),
                               (self.fc3, "node_4")]:
                w_key = f"{prefix}_param1"
                dims = initialisers[w_key]['dims']
                W = np.reshape(initialisers[w_key]['data'], dims, order='F')
                fc.weight.copy_(torch.from_numpy(W))
                b_key = f"{prefix}_param2"
                fc.bias.copy_(torch.from_numpy(initialisers[b_key]['data']))


# ===========================================================================
# Data loading
# ===========================================================================

def load_all_graphs_fortran(folder):
    """Load all graphs from Fortran-exported all_graphs.txt."""
    graph_file = os.path.join(folder, "all_graphs.txt")
    target_file = os.path.join(folder, "first_sample_target.txt")

    # Read normalized targets from Fortran
    # Target normalization is done by Fortran: (y - min) / (max - min)
    # We read raw targets from the Fortran output array (already normalized
    # to [0,1] in all_graphs export... actually no, each graph has just
    # its structure, not its target)
    # We need targets from the 'output' array which was already normalized.

    graphs = []
    with open(graph_file) as f:
        num_samples = int(f.readline().strip())
        for s in range(num_samples):
            header = f.readline().split()
            num_verts = int(header[0])
            num_edges = int(header[1])
            num_csr = int(header[2])
            num_vf = int(header[3])
            num_ef = int(header[4])

            # Vertex features: num_verts rows, each with num_vf values
            vf = np.zeros((num_vf, num_verts), dtype=np.float32)
            for v in range(num_verts):
                vals = [float(x) for x in f.readline().split()]
                vf[:, v] = vals

            # Edge features: num_edges rows, each with num_ef values
            ef = np.zeros((num_ef, num_edges), dtype=np.float32)
            for e in range(num_edges):
                vals = [float(x) for x in f.readline().split()]
                ef[:, e] = vals

            # CSR adj_ia: num_verts + 1 values on one line
            adj_ia = np.array([int(x) for x in f.readline().split()],
                              dtype=np.int64)

            # CSR adj_ja: num_csr rows, each with 2 values
            adj_ja = np.zeros((2, num_csr), dtype=np.int64)
            for j in range(num_csr):
                vals = [int(x) for x in f.readline().split()]
                adj_ja[0, j] = vals[0]
                adj_ja[1, j] = vals[1]

            # read the energy
            target = float(f.readline().strip())

            x, ei, ea, nd = csr_to_pyg(
                torch.from_numpy(vf), torch.from_numpy(ef),
                torch.from_numpy(adj_ia), torch.from_numpy(adj_ja))
            graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, node_degree=nd, y=torch.tensor([target], dtype=torch.float32)))

    # normalise the energies to [0,1] using the min and max from the first_sample_target.txt
    min_energy = min(g.y.item() for g in graphs)
    max_energy = max(g.y.item() for g in graphs)
    for g in graphs:
            g.y = (g.y - min_energy) / (max_energy - min_energy)  # Normalise to [0,1]

    return graphs

# ===========================================================================
# Training
# ===========================================================================

def train(model, loader, lr=1e-2, epochs=20):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    clip_norm = 0.1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for data in loader:
            optimiser.zero_grad()
            out = model(data)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimiser.step()
            total_loss += loss.item()
        # print loss and accuracy
        print(f"Epoch {epoch+1:3d}: Loss = {total_loss / len(loader):.4f} | Accuracy = {100 * (1 - total_loss / len(loader)):.2f}%")

# -------- Main --------
if __name__ == "__main__":
    import random

    seed = 42

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)

    single_thread_cpu = True

    if single_thread_cpu:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.is_available = lambda : False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_file = os.path.join(script_dir, "athena_gnn_init.onnx")

    print("\nLoading training graphs...")
    graphs = load_all_graphs_fortran(script_dir)
    loader = DataLoader(graphs, batch_size=8, shuffle=True)

    initialisers, nodes = parse_athena_onnx(onnx_file)
    model = Net(v_in=6, e_in=1)
    model.load_from_athena_onnx(initialisers)
    # print number of parameters
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    train(model, loader)
