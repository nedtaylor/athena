"""Shared-data rollout trainer for lno_expanded (Python).

This script is intentionally paired with the Fortran trainer:
- both read the same coefficient table,
- both generate trajectories with the same implicit heat solver,
- both train autoregressively on rollout loss,
- both benchmark on the same held-out trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
SHARED_DIR = HERE.parent / 'shared'
COEFF_PATH = SHARED_DIR / 'rollout_coeffs.csv'
FORTRAN_METRICS_PATH = SHARED_DIR / 'fortran_benchmark.txt'
PY_METRICS_PATH = SHARED_DIR / 'python_benchmark.json'
PY_FINAL_STATE_PATH = SHARED_DIR / 'python_final_state.csv'
FIGURES_DIR = HERE / 'figures'


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_coeff_table(path: Path, expected_rows: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sample, c1, c2, c3 = [token.strip() for token in line.split(',')]
            _ = int(sample)
            rows.append([float(c1), float(c2), float(c3)])
    if len(rows) < expected_rows:
        raise ValueError(f'Coefficient rows={len(rows)} but expected at least {expected_rows}')
    return np.asarray(rows[:expected_rows], dtype=np.float32)


def initial_profile(x: np.ndarray, coeffs: np.ndarray, bc_left: float, bc_right: float) -> np.ndarray:
    state = (
        bc_left
        + (bc_right - bc_left) * x
        + coeffs[0] * np.sin(np.pi * x)
        + coeffs[1] * np.sin(2.0 * np.pi * x)
        + coeffs[2] * np.sin(3.0 * np.pi * x)
    ).astype(np.float32)
    state[0] = bc_left
    state[-1] = bc_right
    return state


def build_heat_matrix(grid_size: int, alpha: float, dt: float, dx: float) -> np.ndarray:
    ratio = alpha * dt / max(dx * dx, 1.0e-12)
    matrix = np.zeros((grid_size, grid_size), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[-1, -1] = 1.0
    for idx in range(1, grid_size - 1):
        matrix[idx, idx - 1] = -ratio
        matrix[idx, idx] = 1.0 + 2.0 * ratio
        matrix[idx, idx + 1] = -ratio
    return matrix


def implicit_heat_step(state: np.ndarray, matrix: np.ndarray, bc_left: float, bc_right: float) -> np.ndarray:
    rhs = state.astype(np.float32, copy=True)
    rhs[0] = bc_left
    rhs[-1] = bc_right
    next_state = np.linalg.solve(matrix, rhs).astype(np.float32)
    next_state[0] = bc_left
    next_state[-1] = bc_right
    return next_state


def build_trajectories(coeffs: np.ndarray, config: dict) -> np.ndarray:
    x = np.linspace(0.0, 1.0, config['grid_size'], dtype=np.float32)
    matrix = build_heat_matrix(config['grid_size'], config['alpha'], config['dt'], config['dx'])
    trajectories = np.empty(
        (len(coeffs), config['rollout_benchmark_steps'] + 1, config['grid_size']),
        dtype=np.float32,
    )
    for sample_idx in range(len(coeffs)):
        state = initial_profile(x, coeffs[sample_idx], config['bc_left'], config['bc_right'])
        trajectories[sample_idx, 0] = state
        for step_idx in range(1, config['rollout_benchmark_steps'] + 1):
            state = implicit_heat_step(state, matrix, config['bc_left'], config['bc_right'])
            trajectories[sample_idx, step_idx] = state
    return trajectories


class RolloutMLP(nn.Module):
    def __init__(self, grid_size: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(grid_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, grid_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicLNOLayer(nn.Module):
    """Dynamic Laplace Neural Operator layer matching athena dynamic_lno_layer_type.

    Forward:  v = sigma( D(mu) @ diag(beta) @ E(mu) @ u  +  W @ u  +  b )

    where:
      E[k,j] = exp(-mu_k * t_j),   t_j = (j-1)/(n_in-1)
      D[i,k] = exp(-mu_k * tau_i),  tau_i = (i-1)/(n_out-1)
      mu:   learnable poles        [num_modes]
      beta: learnable residues     [num_modes]
      W:    local bypass weights   [num_outputs, num_inputs]
      b:    bias                   [num_outputs]
    """
    def __init__(self, num_inputs: int, num_outputs: int, num_modes: int,
                 activation: str = 'none') -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_modes = num_modes

        # Learnable poles — initialised to k*pi (matching Fortran)
        self.mu = nn.Parameter(torch.tensor(
            [float(k) * np.pi for k in range(1, num_modes + 1)],
            dtype=torch.float32))
        # Learnable residues
        self.beta = nn.Parameter(torch.zeros(num_modes, dtype=torch.float32))
        # Local bypass weights
        self.W = nn.Parameter(torch.zeros(num_outputs, num_inputs, dtype=torch.float32))
        # Bias
        self.b = nn.Parameter(torch.zeros(num_outputs, dtype=torch.float32))

        # Activation
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'none':
            self.activation = None
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_inputs]
        # Build encoder E [num_modes, num_inputs]
        t = torch.linspace(0.0, 1.0, self.num_inputs, device=x.device, dtype=x.dtype)
        E = torch.exp(-self.mu.unsqueeze(1) * t.unsqueeze(0))  # [M, n_in]

        # Build decoder D [num_outputs, num_modes]
        tau = torch.linspace(0.0, 1.0, self.num_outputs, device=x.device, dtype=x.dtype)
        D = torch.exp(-self.mu.unsqueeze(0) * tau.unsqueeze(1))  # [n_out, M]

        # Spectral path: D @ diag(beta) @ E @ u
        encoded = E @ x.t()  # [M, batch]
        scaled = self.beta.unsqueeze(1) * encoded  # [M, batch]
        spectral = D @ scaled  # [n_out, batch]

        # Local bypass: W @ u
        local = self.W @ x.t()  # [n_out, batch]

        # Combine + bias
        out = spectral + local + self.b.unsqueeze(1)  # [n_out, batch]
        out = out.t()  # [batch, n_out]

        if self.activation is not None:
            out = self.activation(out)
        return out


class RolloutLNO(nn.Module):
    """Two-layer network using DynamicLNO layers, matching the Fortran architecture."""
    def __init__(self, grid_size: int, hidden: int, num_modes: int) -> None:
        super().__init__()
        self.layer1 = DynamicLNOLayer(grid_size, hidden, num_modes, activation='relu')
        self.layer2 = DynamicLNOLayer(hidden, grid_size, num_modes, activation='none')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def _lcg_next(state: int) -> int:
    return (1103515245 * state + 12345) % 2147483647


def _apply_shared_init_mlp(model: RolloutMLP, seed: int, scale: float = 0.05) -> None:
    state = int(seed)
    with torch.no_grad():
        layer1 = model.net[0]
        layer2 = model.net[2]

        for in_idx in range(layer1.in_features):
            for out_idx in range(layer1.out_features):
                state = _lcg_next(state)
                value = np.float32(scale * (2.0 * (state / 2147483647.0) - 1.0))
                layer1.weight[out_idx, in_idx] = torch.tensor(value, dtype=layer1.weight.dtype)

        for out_idx in range(layer1.out_features):
            state = _lcg_next(state)
            value = np.float32(scale * (2.0 * (state / 2147483647.0) - 1.0))
            layer1.bias[out_idx] = torch.tensor(value, dtype=layer1.bias.dtype)

        for in_idx in range(layer2.in_features):
            for out_idx in range(layer2.out_features):
                state = _lcg_next(state)
                value = np.float32(scale * (2.0 * (state / 2147483647.0) - 1.0))
                layer2.weight[out_idx, in_idx] = torch.tensor(value, dtype=layer2.weight.dtype)

        for out_idx in range(layer2.out_features):
            state = _lcg_next(state)
            value = np.float32(scale * (2.0 * (state / 2147483647.0) - 1.0))
            layer2.bias[out_idx] = torch.tensor(value, dtype=layer2.bias.dtype)


def _apply_shared_init_lno(model: RolloutLNO, seed: int, scale: float = 0.05) -> None:
    """Shared initialisation for LNO model matching Fortran dynamic_lno_layer_type.

    For each LNO layer, the Fortran init sets:
      - mu[k] = k * pi           (poles, NOT from LCG)
      - beta[k] from LCG         (residues)
      - W from LCG               (bypass weights, column-major order)
      - b from LCG               (bias)
    """
    state = int(seed)
    with torch.no_grad():
        for layer in [model.layer1, model.layer2]:
            # mu: poles initialised to k*pi (not from LCG, matching Fortran)
            for k in range(layer.num_modes):
                layer.mu[k] = float(k + 1) * np.pi

            # beta: residues from LCG
            for k in range(layer.num_modes):
                state = _lcg_next(state)
                value = np.float32(scale * (2.0 * (state / 2147483647.0) - 1.0))
                layer.beta[k] = torch.tensor(value, dtype=layer.beta.dtype)

            # W: bypass weights from LCG (column-major: iterate in_idx then out_idx)
            for in_idx in range(layer.num_inputs):
                for out_idx in range(layer.num_outputs):
                    state = _lcg_next(state)
                    value = np.float32(scale * (2.0 * (state / 2147483647.0) - 1.0))
                    layer.W[out_idx, in_idx] = torch.tensor(value, dtype=layer.W.dtype)

            # b: bias from LCG
            for out_idx in range(layer.num_outputs):
                state = _lcg_next(state)
                value = np.float32(scale * (2.0 * (state / 2147483647.0) - 1.0))
                layer.b[out_idx] = torch.tensor(value, dtype=layer.b.dtype)


def apply_shared_initialization(model: nn.Module, seed: int, scale: float = 0.05) -> None:
    if isinstance(model, RolloutLNO):
        _apply_shared_init_lno(model, seed, scale)
    else:
        _apply_shared_init_mlp(model, seed, scale)


def rollout_loss(model: nn.Module, initial_state: torch.Tensor, target_traj: torch.Tensor, steps: int, bc_left: float, bc_right: float) -> torch.Tensor:
    state = initial_state
    loss = 0.0
    for step in range(1, steps + 1):
        state = model(state)
        state = state.clone()
        state = torch.clamp(state, min=-4.0, max=4.0)
        state[:, 0] = bc_left
        state[:, -1] = bc_right
        loss = loss + F.mse_loss(state, target_traj[:, step, :])
    return loss / float(steps)


def train_rollout(model: nn.Module, train_traj: np.ndarray, val_traj: np.ndarray, config: dict) -> None:
    print('\n' + '=' * 60)
    print('STEP 1 / 2 : Rollout Training (Python)')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_tensor = torch.from_numpy(train_traj).to(device)
    val_tensor = torch.from_numpy(val_traj).to(device)

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss = 0.0
        for idx in range(train_tensor.shape[0]):
            batch_traj = train_tensor[idx:idx + 1]
            state = batch_traj[:, 0, :]
            optimizer.zero_grad()
            loss = 0.0
            for step in range(1, config['rollout_train_steps'] + 1):
                pred = model(state)
                target = batch_traj[:, step, :]
                loss += F.mse_loss(pred, target)
                state = torch.clamp(pred, -4.0, 4.0)
                state = state.clone()
                state[:, 0] = config['bc_left']
                state[:, -1] = config['bc_right']

            loss = loss / config['rollout_train_steps']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.inference_mode():
            val_loss = 0.0
            for idx in range(val_tensor.shape[0]):
                batch_traj = val_tensor[idx:idx + 1]
                val_loss += rollout_loss(
                    model,
                    batch_traj[:, 0, :],
                    batch_traj,
                    config['rollout_train_steps'],
                    config['bc_left'],
                    config['bc_right'],
                ).item()

        train_loss /= max(train_tensor.shape[0] * config['rollout_train_steps'], 1)
        val_loss /= max(val_tensor.shape[0], 1)
        print(f"Epoch {epoch:02d} | train={train_loss:.6f} | val={val_loss:.6f}")


def run_neural_rollout(model: nn.Module, start_state: np.ndarray, config: dict) -> np.ndarray:
    device = next(model.parameters()).device
    state = torch.from_numpy(start_state).unsqueeze(0).to(device)
    history = [start_state.copy()]
    model.eval()
    with torch.inference_mode():
        for _ in range(config['rollout_benchmark_steps']):
            state = model(state)
            state = state.clone()
            state = torch.clamp(state, min=-4.0, max=4.0)
            state[:, 0] = config['bc_left']
            state[:, -1] = config['bc_right']
            history.append(state.squeeze(0).cpu().numpy().copy())
    return np.asarray(history, dtype=np.float32)


def parse_fortran_metrics(path: Path) -> tuple[dict[str, float], dict[str, float]] | None:
    if not path.exists():
        return None
    parsed: dict[str, float] = {}
    refs: dict[str, float] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, raw_value = line.split('=', 1)
            if ',ref=' in raw_value:
                pred_value, ref_value = raw_value.split(',ref=', 1)
                parsed[key] = float(pred_value)
                refs[key.replace('_pred', '_ref')] = float(ref_value)
            else:
                parsed[key] = float(raw_value)
    return parsed, refs


def benchmark(model: nn.Module, benchmark_traj: np.ndarray, config: dict) -> dict[str, float]:
    print('\n' + '=' * 60)
    print('STEP 2 / 2 : Shared Benchmark')
    print('=' * 60)

    neural_history = run_neural_rollout(model, benchmark_traj[0], config)
    ref_history = benchmark_traj

    neural_final = neural_history[-1]
    ref_final = ref_history[-1]
    rel_error_pct = float(np.linalg.norm(neural_final - ref_final) / (np.linalg.norm(ref_final) + 1.0e-12) * 100.0)
    max_abs_error = float(np.max(np.abs(neural_final - ref_final)))

    metrics = {
        'python_rel_error_pct': rel_error_pct,
        'python_max_abs_error': max_abs_error,
        'python_final_checksum': float(np.sum(neural_final)),
        'reference_final_checksum': float(np.sum(ref_final)),
    }

    fortran_data = parse_fortran_metrics(FORTRAN_METRICS_PATH)
    if fortran_data is not None:
        fortran_metrics, fortran_refs = fortran_data
        py_key_vals = [float(v) for v in neural_final]
        ft_key_vals = []
        for idx in range(1, config['grid_size'] + 1):
            key = f'final_state_{idx}_pred'
            if key in fortran_metrics:
                ft_key_vals.append(fortran_metrics[key])
        if len(ft_key_vals) == config['grid_size']:
            delta = np.max(np.abs(np.asarray(py_key_vals) - np.asarray(ft_key_vals)))
            metrics['python_vs_fortran_pred_max_abs'] = float(delta)

        ft_ref_vals = []
        for idx in range(1, config['grid_size'] + 1):
            key = f'final_state_{idx}_ref'
            if key in fortran_refs:
                ft_ref_vals.append(fortran_refs[key])
        if len(ft_ref_vals) == config['grid_size']:
            ref_delta = np.max(np.abs(ref_final - np.asarray(ft_ref_vals, dtype=np.float32)))
            metrics['python_vs_fortran_ref_max_abs'] = float(ref_delta)

    x = np.linspace(0.0, 1.0, config['grid_size'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, ref_final, label='Reference rollout', lw=2)
    axes[0].plot(x, neural_final, '--', label='Python surrogate', lw=2)
    axes[0].set_title('Final rollout state')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Temperature [K]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    step_errors = []
    for step in range(1, len(neural_history)):
        err = np.linalg.norm(neural_history[step] - ref_history[step])
        norm = np.linalg.norm(ref_history[step]) + 1.0e-12
        step_errors.append(err / norm * 100.0)
    axes[1].plot(np.arange(1, len(step_errors) + 1), step_errors, lw=2)
    axes[1].set_title('Rollout relative error')
    axes[1].set_xlabel('Rollout step')
    axes[1].set_ylabel('Error [%]')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'benchmark_rollout.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    with PY_METRICS_PATH.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    np.savetxt(PY_FINAL_STATE_PATH, neural_final, delimiter=',', fmt='%.10f')

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {PY_METRICS_PATH}")
    print(f"Saved final state to {PY_FINAL_STATE_PATH}")
    return metrics


def build_config(args: argparse.Namespace) -> dict:
    return {
        'grid_size': args.grid_size,
        'n_samples': args.n_samples,
        'n_train': args.n_train,
        'n_val': args.n_val,
        'benchmark_idx': args.benchmark_idx,
        'rollout_train_steps': args.rollout_train_steps,
        'rollout_benchmark_steps': args.rollout_benchmark_steps,
        'alpha': args.alpha,
        'dt': args.dt,
        'bc_left': args.bc_left,
        'bc_right': args.bc_right,
        'epochs': args.epochs,
        'hidden': args.hidden,
        'lr': args.lr,
        'dx': 1.0 / float(args.grid_size - 1),
        'seed': args.seed,
        'model': args.model,
        'num_modes': args.num_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Shared rollout trainer for lno_expanded (Python).')
    parser.add_argument('--grid_size', type=int, default=48)
    parser.add_argument('--n_samples', type=int, default=24)
    parser.add_argument('--n_train', type=int, default=16)
    parser.add_argument('--n_val', type=int, default=4)
    parser.add_argument('--benchmark_idx', type=int, default=21)
    parser.add_argument('--rollout_train_steps', type=int, default=4)
    parser.add_argument('--rollout_benchmark_steps', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=1.0e-2)
    parser.add_argument('--dt', type=float, default=8.0e-4)
    parser.add_argument('--bc_left', type=float, default=-1.0)
    parser.add_argument('--bc_right', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1.0e-4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--model', type=str, default='lno', choices=['mlp', 'lno'],
                        help='Model architecture: mlp or lno')
    parser.add_argument('--num_modes', type=int, default=16,
                        help='Number of Laplace spectral modes (LNO only)')
    args = parser.parse_args()

    ensure_dirs()
    config = build_config(args)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    coeffs = parse_coeff_table(COEFF_PATH, config['n_samples'])
    trajectories = build_trajectories(coeffs, config)

    train_traj = trajectories[:config['n_train']]
    val_traj = trajectories[config['n_train']:config['n_train'] + config['n_val']]
    benchmark_traj = trajectories[config['benchmark_idx'] - 1]

    if config.get('model', 'mlp') == 'lno':
        model = RolloutLNO(config['grid_size'], config['hidden'], config['num_modes'])
        print(f'Model: RolloutLNO (modes={config["num_modes"]}, '
              f'params={sum(p.numel() for p in model.parameters())})')
    else:
        model = RolloutMLP(config['grid_size'], config['hidden'])
        print(f'Model: RolloutMLP (params={sum(p.numel() for p in model.parameters())})')
    apply_shared_initialization(model, config['seed'])
    train_rollout(model, train_traj, val_traj, config)
    benchmark(model, benchmark_traj, config)


if __name__ == '__main__':
    main()
