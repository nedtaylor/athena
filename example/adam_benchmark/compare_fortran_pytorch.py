# %% [markdown]
# # Fortran vs PyTorch Adam — compact comparison
# Builds the Fortran benchmark, runs PyTorch Adam on the same problems,
# then returns per-step absolute differences.

# %% Imports & hyperparameters
import pathlib
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

LR      = np.float32(0.01)
BETA1   = np.float32(0.9)
BETA2   = np.float32(0.999)
EPS     = np.float32(1e-8)
N_STEPS = 20
X0_SCALAR = np.array([0.0], dtype=np.float32)
X0_MULTI  = np.array([0.0, 2.0], dtype=np.float32)

# %% Build & run Fortran benchmark
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
result = subprocess.run(
    ["fpm", "run", "benchmark_adam", "--example"],
    cwd=str(REPO_ROOT), capture_output=True, text=True,
)
if result.returncode != 0:
    raise RuntimeError(result.stderr[-3000:])
print(next((l for l in reversed(result.stdout.splitlines()) if l.strip()), ""))

SCALAR_CSV = REPO_ROOT / "fortran_adam_scalar.csv"
MULTI_CSV  = REPO_ROOT / "fortran_adam_multi.csv"

def _load(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if c != "step":
            df[c] = df[c].astype(np.float32)
    return df

df_scalar = _load(SCALAR_CSV)
df_multi  = _load(MULTI_CSV)

# %% Run PyTorch Adam
def _run_pytorch(x0, loss_fn, n=N_STEPS):
    x = torch.tensor(x0.copy(), dtype=torch.float32).requires_grad_(True)
    opt = optim.Adam([x], lr=float(LR), betas=(float(BETA1), float(BETA2)), eps=float(EPS))
    history = []
    for step in range(1, n + 1):
        opt.zero_grad()
        loss = loss_fn(x)            # loss BEFORE the step (matches Fortran)
        loss.backward()
        opt.step()
        state = opt.state[x]
        history.append(dict(
            step=step,
            param=x.detach().numpy().astype(np.float32).copy(),
            m=state["exp_avg"].detach().numpy().astype(np.float32).copy(),
            v=state["exp_avg_sq"].detach().numpy().astype(np.float32).copy(),
            loss=np.float32(loss.item()),
        ))
    return history

pt_scalar = _run_pytorch(X0_SCALAR, lambda x: (x[0] - 3.0) ** 2)
pt_multi  = _run_pytorch(X0_MULTI,  lambda x: (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2)

# %% Compare: scalar problem
def compare_scalar(pt_hist, df):
    rows = []
    for pt, (_, f) in zip(pt_hist, df.iterrows()):
        rows.append(dict(
            step  = int(pt["step"]),
            param = float(pt["param"][0]),
            d_param = abs(float(pt["param"][0]) - float(f["param_1"])),
            d_m     = abs(float(pt["m"][0])     - float(f["m_1"])),
            d_v     = abs(float(pt["v"][0])     - float(f["v_1"])),
            d_loss  = abs(float(pt["loss"])     - float(f["loss"])),
        ))
    deltas = pd.DataFrame(rows).set_index("step")
    print("Scalar — max absolute differences (PyTorch vs Fortran):")
    print(deltas[["d_param", "d_m", "d_v", "d_loss"]].max().to_string())
    return deltas

deltas_scalar = compare_scalar(pt_scalar, df_scalar)

# %% Compare: multi-parameter problem
def compare_multi(pt_hist, df):
    rows = []
    for pt, (_, f) in zip(pt_hist, df.iterrows()):
        rows.append(dict(
            step    = int(pt["step"]),
            d_p1    = abs(float(pt["param"][0]) - float(f["param_1"])),
            d_p2    = abs(float(pt["param"][1]) - float(f["param_2"])),
            d_m1    = abs(float(pt["m"][0])     - float(f["m_1"])),
            d_m2    = abs(float(pt["m"][1])     - float(f["m_2"])),
            d_v1    = abs(float(pt["v"][0])     - float(f["v_1"])),
            d_v2    = abs(float(pt["v"][1])     - float(f["v_2"])),
            d_loss  = abs(float(pt["loss"])     - float(f["loss"])),
        ))
    deltas = pd.DataFrame(rows).set_index("step")
    print("Multi — max absolute differences (PyTorch vs Fortran):")
    print(deltas.max().to_string())
    return deltas

deltas_multi = compare_multi(pt_multi, df_multi)
