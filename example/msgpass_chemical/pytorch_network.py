import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # replaces deprecated `data.DataLoader`
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from ase.neighborlist import neighbor_list
from ase.io import read
import os
import numpy as np
from glob import glob
from torch_geometric.nn import global_add_pool

# -------- Duvenaud-style Message Passing Layer --------
class DuvenaudMPNN(MessagePassing):
    def __init__(self, in_channels, edge_channels, max_degree, min_degree=1, num_timesteps=4, num_outputs=10):
        super().__init__(aggr='add')  # 'add' for sum aggregation
        self.num_timesteps = num_timesteps
        self.max_degree = max_degree
        self.min_degree = min_degree
        if min_degree < 1 or max_degree < min_degree:
            raise ValueError("min_degree must be at least 1 and max_degree must be at least min_degree.")
        self.mlps = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_channels + edge_channels, in_channels, bias=False),
                    nn.Sigmoid()
                )
                for _ in range(min_degree, max_degree + 1)
            ])
            for _ in range(num_timesteps)
        ])
        self.readout = nn.ModuleList()
        for t in range(num_timesteps):
            self.readout.append(nn.Sequential(
                nn.Linear(in_channels, num_outputs, bias=False),
                nn.Softmax(dim=1)  # Softmax for classification tasks
            ))

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Add dummy edge attributes for self-loops
        num_edges_added = num_nodes  # since one self-loop per node
        loop_attr = torch.zeros(num_edges_added, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        # Now compute degrees
        src_degree = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        edge_degree = src_degree[edge_index[0]].clamp(max=self.max_degree)

        for t in range(self.num_timesteps):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr, deg=edge_degree, t=torch.full_like(edge_degree, t))

        # Readout
        out = [self.readout[t](x) for t in range(self.num_timesteps)]
        out = torch.stack(out, dim=1).sum(dim=1)
        return out

    def message(self, x_j, edge_attr, deg, t):
        """
        x_j: neighbour node features [num_edges, in_channels]
        edge_attr: edge features [num_edges, edge_channels]
        deg: degree bucket for each edge's source node
        t: current timestep, broadcasted to shape [num_edges]
        """
        msg_input = torch.cat([x_j, edge_attr], dim=-1)

        # Prepare output tensor
        out = torch.zeros_like(x_j)

        # For each degree class (bucket), apply the corresponding MLP
        for d in range(self.min_degree, self.max_degree + 1):
            idx = (deg == d)
            if idx.any():
                mlp = self.mlps[t[0].item()][d - self.min_degree]
                out[idx] = mlp(msg_input[idx])
        return out

# -------- Network Definition --------
class Net(nn.Module):
    def __init__(self, v_in, e_in, num_outputs=1):
        super().__init__()
        # self.mpnn1 = MFConv(v_in, v_in, 10, bias=False)
        self.mpnn1 = DuvenaudMPNN(v_in, e_in, 10, 1, 4, 10)
        # self.fc_readout = nn.Linear(v_in, 10, bias=False)
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        x = self.mpnn1(data.x, data.edge_index, data.edge_attr)
        x = global_add_pool(x, data.batch)  # Aggregate node features per graph
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x


# -------- Dummy Graph from ASE .xyz File --------
def ase_to_graph(file):
    atoms_list = read(file, index=":")
    print(f"Processing {file}: {len(atoms_list)} structures found.")
    data_list = []
    for atoms in atoms_list:
        # print(f"  - {len(atoms)} atoms in structure.")
        Z = atoms.get_atomic_numbers() / 100.0
        mass = atoms.get_masses() / 52.0
        pos = atoms.get_positions()
        forces = atoms.get_forces()


        cutoff = 3.0
        # Use ASE's neighbour list to find all neighbours within cutoff (with periodic images)
        i, j, S = neighbor_list('ijS', atoms, cutoff=cutoff)

        # Compute the displacements (accounting for periodic images)
        cell = atoms.get_cell()
        displacement = pos[j] + S @ cell - pos[i]
        edge_length = torch.tensor((displacement**2).sum(axis=1)**0.5, dtype=torch.float32)

        # Edge features: distance
        edge_attr = edge_length.view(-1, 1)

        # Construct PyTorch Geometric data
        edge_index = torch.tensor(np.array([i, j]), dtype=torch.long)
        # get degree of each node and use it as a node feature
        deg = degree(edge_index[0], num_nodes=len(atoms), dtype=torch.float)

        # Create node features
        x = torch.tensor(np.column_stack((forces, Z, mass, deg)), dtype=torch.float)
        y = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    # renormalise all y values by converting to 0-1 range
    if data_list:
        y_min = min(data.y.item() for data in data_list)
        y_max = max(data.y.item() for data in data_list)
        for data in data_list:
            data.y = (data.y - y_min) / (y_max - y_min)
    return data_list

# -------- Load Dataset --------
def load_dataset(folder):
    # return a concatenated list of graphs from all .xyz files in the folder
    data_list = []
    for file in sorted(glob(os.path.join(folder, "*.xyz"))):
        graphs = ase_to_graph(file)
        data_list.extend(graphs)
    print(f"Loaded {len(data_list)} graphs from {folder}.")
    return data_list

# -------- Training --------
def train(model, loader, lr=1e-2, epochs=100):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for data in loader:
            optimiser.zero_grad()
            out = model(data)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        # print loss and accuracy
        print(f"Epoch {epoch+1:3d}: Loss = {total_loss / len(loader):.4f} | Accuracy = {100 * (1 - total_loss / len(loader)):.2f}%")

# -------- Main --------
if __name__ == "__main__":
    import random

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    # torch.use_deterministic_algorithms(True)

    torch.set_default_dtype(torch.float32)
    dataset = load_dataset(".")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = Net(v_in=6, e_in=1)
    # print number of parameters
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    train(model, loader)
