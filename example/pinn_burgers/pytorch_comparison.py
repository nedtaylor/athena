# # Physics-Informed Neural Network (PINN) for 1D Burgers Equation
# This script implements a PINN to solve the 1D Burgers equation using PyTorch.
# This has been adapted from an online tutorial: https://www.marktechpost.com/2025/03/28/a-step-by-step-guide-to-solve-1d-burgers-equation-with-physics-informed-neural-networks-pinns-a-pytorch-approach-using-automatic-differentiation-and-collocation-methods/
# The code can optionally read initial parameter values and data from athena output files for comparison.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# %%
input_from_athena = True # set to True to read parameter initial values and data from athena output files
output_from_athena = False # set to True to plot the results from athena output files

# %%
precision = torch.float32
torch.set_default_dtype(precision)

# set the random number seed
torch.manual_seed(42)


# %% Initialise the data

x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0
nu = 0.01 / np.pi

N_f = 1000
N_0 = 200
N_b = 200

if input_from_athena:
    X_f = np.loadtxt("../../X_f.txt")
else:
    X_f = np.random.rand(N_f, 2)
    X_f[:, 0] = X_f[:, 0] * (x_max - x_min) + x_min  # x in [-1, 1]
    X_f[:, 1] = X_f[:, 1] * (t_max - t_min) + t_min    # t in [0, 1]

x0 = np.linspace(x_min, x_max, N_0)[:, None]
t0 = np.zeros_like(x0)
u0 = -np.sin(np.pi * x0)

tb = np.linspace(t_min, t_max, N_b)[:, None]
xb_left = np.ones_like(tb) * x_min
xb_right = np.ones_like(tb) * x_max
ub_left = np.zeros_like(tb)
ub_right = np.zeros_like(tb)

X_f = torch.tensor(X_f, dtype=precision, requires_grad=True)
x0 = torch.tensor(x0, dtype=precision)
t0 = torch.tensor(t0, dtype=precision)
u0 = torch.tensor(u0, dtype=precision)
tb = torch.tensor(tb, dtype=precision)
xb_left = torch.tensor(xb_left, dtype=precision)
xb_right = torch.tensor(xb_right, dtype=precision)
ub_left = torch.tensor(ub_left, dtype=precision)
ub_right = torch.tensor(ub_right, dtype=precision)


# %% Initialise the PINN model

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()

        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1], bias=True ))
        self.layers = nn.ModuleList(layer_list)
        # initialise the parameters
        if input_from_athena:
          # set the parameters from a read in file text file, where each line is a separate parameter
          param_file = "../../params.txt"
          with open(param_file, "r") as f:
              params = f.readlines()
          for i, layer in enumerate(self.layers):
              layer.weight.data = torch.tensor([float(x) for x in params[i*2].split()]).view(layer.weight.data.shape)
              tmp_data = torch.tensor([float(x) for x in params[i*2].split()]).view(layer.weight.data.shape[::-1])
              layer.weight.data = torch.tensor(torch.transpose(tmp_data, 0, 1), dtype=precision) #.view(layer.weight.data.shape)
              layer.bias.data = torch.tensor([float(x) for x in params[i*2+1].split()]).view(layer.bias.data.shape)
        else:
          for layer in self.layers:
              nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="tanh")
              layer.bias.data.fill_(1)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)

layers = [2, 50, 50, 50, 50, 1]
model = PINN(layers)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# %% Define the loss function

def pde_residual(model, X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    u = model(torch.cat([x, t], dim=1))
    torch.set_printoptions(precision=20, sci_mode=False)
    np.set_printoptions(precision=20, floatmode="unique")

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    f = u_t + u * u_x - nu * u_xx
    return f

def loss_func(model):
    f_pred = pde_residual(model, X_f.to(device))
    loss_f = torch.mean(f_pred**2)

    u0_pred = model(torch.cat([x0.to(device), t0.to(device)], dim=1))
    loss_0 = torch.mean((u0_pred - u0.to(device))**2)

    u_left_pred = model(torch.cat([xb_left.to(device), tb.to(device)], dim=1))
    u_right_pred = model(torch.cat([xb_right.to(device), tb.to(device)], dim=1))
    loss_b = torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2)

    torch.set_printoptions(precision=10, sci_mode=False)
    np.set_printoptions(precision=10, floatmode="unique")
    loss = loss_f + loss_0 + loss_b
    return loss

# %% Set up optimiser

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
num_epochs = 100
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total number of parameters", pytorch_total_params)

# %% Train the model

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_func(model)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.5e}')

print("Training complete!")


# %% Plot the results

N_x, N_t = 256, 100
if output_from_athena:
  fort_data = np.loadtxt("../../u_pred.txt")
  X = fort_data[:, 0].reshape(N_t, N_x)
  T = fort_data[:, 1].reshape(N_t, N_x)
else:
  x = np.linspace(x_min, x_max, N_x)
  t = np.linspace(t_min, t_max, N_t)
  X, T = np.meshgrid(x, t)
XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
XT_tensor = torch.tensor(XT, dtype=torch.float32).to(device)

if output_from_athena:
    # read file from u_pred.txt as X, T, u_pred
    u_pred = fort_data[:, 2].reshape(N_t, N_x)
    label = "athena"
else:
    model.eval()
    with torch.no_grad():
        u_pred = model(XT_tensor).cpu().numpy().reshape(N_t, N_x)
    label = "pytorch"


plt.figure(figsize=(8, 5))
plt.contourf(X, T, u_pred, levels=100, cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title("Predicted solution u(x,t) via PINN " + label)
plt.show()
