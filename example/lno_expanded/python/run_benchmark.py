"""
Benchmark Runner: Neural LNO vs Pure FDM
=========================================

Runs side-by-side comparisons of the trained Cattaneo-LNO model against
a pure FDM solver for each inference mode:

  1. Hybrid solver   – neural surrogate with FDM fallback vs pure FDM
  2. Super-resolution – coarse→fine neural prediction vs fine-grid FDM
  3. Frame generation – dense temporal frames via neural sub-stepping vs FDM
  4. Warm-start      – neural warm-start + FDM refinement vs cold-start FDM

Each benchmark produces:
  - Console summary (timing, accuracy)
  - Comparison figures saved to figures/
  - JSON results saved to results/

Usage:
    python run_benchmark.py                     # run all benchmarks
    python run_benchmark.py --mode hybrid       # run one benchmark
    python run_benchmark.py --mode super-res --num_steps 200
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import splu

from cattaneo_lno import create_cattaneo_model
from evaluate import load_model
from HF_Cattaneo import (nl_solve_HF_1d_Cattaneo, sparse_matrix_1d,
                          _load_material)
from hybrid_solver import HybridCattaneoSolver, HybridSolverConfig
from inference_modes import (NeuralWarmStart, SuperResolutionInference,
                             TemporalFrameGenerator)


# ── Default physical parameters ───────────────────────────────────────
# alpha, rho_cp, and timestep_jump are overwritten after model load
# to match the material file and training config exactly.
DEFAULTS = dict(
    grid_size=112,
    dx=1e-8,
    dt=1e-13,
    tau=1e-9,
    alpha=1e-4,        # placeholder — overwritten by _sync_physics()
    rho_cp=1e6,        # placeholder — overwritten by _sync_physics()
    bc_left=100.0,
    bc_right=200.0,
    timestep_jump=1,   # placeholder — overwritten by _sync_physics()
    num_steps=0,       # when <= 0, derive steps from time_over_tau
    time_over_tau=50.0,
    spectral_filter='exponential',
    filter_strength=4.0,
    checkpoint='checkpoints/best_model.pt',
)


def _resolve_num_steps(args, timestep_jump: int, default_time_over_tau: float) -> int:
    """Resolve benchmark horizon from either raw steps or physical time.

    NOTE: args.dt is the *effective* dt stored in the training data, which
    already includes the timestep_jump factor (dt_stored = dt_base * TJ).
    We do NOT multiply by timestep_jump again.
    """
    if getattr(args, 'num_steps', 0) > 0:
        return int(args.num_steps)

    time_over_tau = float(getattr(args, 'time_over_tau', default_time_over_tau))
    if time_over_tau <= 0:
        time_over_tau = default_time_over_tau

    dt_eff = args.dt  # already includes timestep_jump
    total_time = time_over_tau * args.tau
    return max(1, int(np.ceil(total_time / max(dt_eff, 1e-30))))


def _get_material_constants(T_ref: float = 200.0, material_file: int = 1):
    """Derive alpha and rho_cp from the material data file at T_ref.

    This mirrors the logic in CattaneoDataGenerator so that benchmarks
    use exactly the same physical constants as the training data.
    """
    from HF_Cattaneo import _load_material
    mat = _load_material(material_file)
    T_grid = mat['T_grid']
    slope = (len(T_grid) - 1) / (mat['T_max'] - mat['T_min'])
    idx = int((T_ref - mat['T_min']) * slope)
    idx = max(0, min(len(T_grid) - 2, idx))
    t = (T_ref - T_grid[idx]) / (T_grid[idx + 1] - T_grid[idx])
    Cv = float(mat['hc_values'][idx] * (1 - t) + mat['hc_values'][idx + 1] * t)
    k = float(mat['tk_values'][idx] * (1 - t) + mat['tk_values'][idx + 1] * t)
    return k / Cv, Cv  # alpha, rho_cp


def _sync_physics(args, model, ckpt):
    """Override args with values that were actually used during training.

    Reads alpha, rho_cp from the material file, and dt/timestep_jump
    from the checkpoint config or training data, so the benchmark runs
    at exactly the same Fourier number as training.
    """
    alpha, rho_cp = _get_material_constants()
    args.alpha = alpha
    args.rho_cp = rho_cp
    args.timestep_jump = getattr(model, 'timestep_jump', args.timestep_jump)

    # ── Recover the training dt ──
    # The training data may use a different dt from the CLI default.
    # Try: checkpoint config → training data file → leave as-is.
    training_dt = None
    cfg = ckpt.get('config', {})
    if 'training_dt' in cfg:
        training_dt = cfg['training_dt']
    if training_dt is None:
        # Fall back: read dt directly from training data
        data_path = Path('data/training_data.pt')
        if data_path.exists():
            data = torch.load(data_path, map_location='cpu', weights_only=False)
            if isinstance(data, dict) and 'dt' in data:
                training_dt = data['dt']
            elif isinstance(data, dict):
                for split in ('train', 'val'):
                    if split in data and isinstance(data[split], dict) and 'dt' in data[split]:
                        training_dt = data[split]['dt']
                        break
    if training_dt is not None and training_dt != args.dt:
        print(f"  Physics sync: overriding dt {args.dt:.2e} → {training_dt:.2e} (from training data)")
        args.dt = training_dt

    print(f"  Physics sync: alpha={alpha:.6e}, rho_cp={rho_cp:.6e}, "
          f"timestep_jump={args.timestep_jump}, dt={args.dt:.2e}")


def _load(args):
    """Load trained model from checkpoint."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_model(
        args.checkpoint, args.grid_size, device, dx=args.dx,
        spectral_filter=args.spectral_filter,
        filter_strength=args.filter_strength,
        use_ghost_cells=True,
    )
    _sync_physics(args, model, ckpt)

    return model, device


def _make_T0(grid_size, bc_left, bc_right):
    """Non-equilibrium initial condition satisfying both boundary values.

    Linear baseline between BCs plus a sinusoidal perturbation that
    vanishes at the boundaries, giving visible thermal evolution.
    """
    x_norm = np.linspace(0, 1, grid_size)
    T0 = bc_left + (bc_right - bc_left) * x_norm
    T0 += 1.2 * (bc_right - bc_left) * np.sin(np.pi * x_norm)
    return T0


def _cuda_sync():
    """Synchronize CUDA before timing to get accurate measurements."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _warm_up(model, device, grid_size, args):
    """Run a few dummy forward passes to warm up CUDA kernels."""
    T_dummy = np.linspace(args.bc_left, args.bc_right, grid_size)
    with torch.inference_mode():
        T_t = torch.from_numpy(T_dummy).float().unsqueeze(0).to(device)
        T_prev_t = T_t.clone()
        q_t = torch.zeros(1, grid_size, device=device)
        tau_t = torch.full((1, grid_size), args.tau, device=device)
        alpha_t = torch.full((1, grid_size), args.alpha, device=device)
        rho_cp_t = torch.full((1, grid_size), args.rho_cp, device=device)
        bc_l = torch.tensor([args.bc_left], device=device)
        bc_r = torch.tensor([args.bc_right], device=device)
        dt_eff = args.dt  # already includes timestep_jump
        for _ in range(10):
            model(T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                  bc_l, bc_r, dt_eff, args.dx)
    _cuda_sync()


def _run_pure_fdm(T0, grid_size, dx, dt, tau, num_fdm_steps,
                  bc_left, bc_right, save_every=1):
    """Fast direct-solve FDM for constant-coefficient Cattaneo equation.

    Pre-factors the time-invariant system matrix once with SuperLU, then
    solves each timestep with O(n) back-substitution instead of full
    Newton-Krylov iterations.  Gives identical results to the iterative
    solver for const=True, k/c not temperature-dependent.
    """
    grid = np.ones(grid_size, dtype=int)

    # ── Build constant HA, BA ──
    B1 = [bc_left, bc_right]
    TB1 = [bc_left, bc_right]
    HA, BA = sparse_matrix_1d(grid_size, grid, T0, B1, TB1,
                              dx=dx, const=True, k_temp_dependent=False)

    # ── Constant heat capacity A (at T_ref=200 K), G=0 ──
    A_vec = np.empty(grid_size, dtype=np.float64)
    for file_id in np.unique(grid):
        mat = _load_material(file_id)
        mask = grid == file_id
        T_ref = 200.0
        T_grid = mat['T_grid']
        hc = mat['hc_values']
        slope = (len(T_grid) - 1) / (mat['T_max'] - mat['T_min'])
        idx = int((T_ref - mat['T_min']) * slope)
        idx = max(0, min(len(T_grid) - 2, idx))
        t = (T_ref - T_grid[idx]) / (T_grid[idx + 1] - T_grid[idx])
        A_vec[mask] = float(hc[idx] * (1 - t) + hc[idx + 1] * t)

    # ── Constant gamma (G=0 ⇒ no dCv/dT terms) ──
    dt_inv = 1.0 / dt
    dt2_inv = 1.0 / (dt * dt)
    gamma = A_vec * dt_inv + tau * A_vec * dt2_inv   # constant vector

    # ── Pre-factor system matrix M = diag(gamma) - HA ──
    M = sp_diags(gamma, format='csc') - HA
    LU = splu(M.tocsc())

    # ── Time-stepping (direct solve each step) ──
    T = T0.copy()
    T_prev = T0.copy()
    history = [T.copy()]

    t0 = time.perf_counter()
    for step in range(num_fdm_steps):
        # omega = -A*T/dt + tau*A*(T_prev - 2*T)/dt²   (G=0 simplification)
        omega = -A_vec * T * dt_inv + tau * A_vec * (T_prev - 2.0 * T) * dt2_inv
        rhs = BA - omega
        T_new = LU.solve(rhs)

        T_prev = T
        T = T_new
        if (step + 1) % save_every == 0:
            history.append(T.copy())

    elapsed = time.perf_counter() - t0
    return np.array(history), elapsed


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 0: Autoregressive rollout vs pure FDM
# ═══════════════════════════════════════════════════════════════════════

def benchmark_rollout(args):
    """Compare long-horizon temperature evolution against pure FDM."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Rollout Temperature Evolution  (Neural  vs  Pure FDM)")
    print("=" * 70)

    model, device = _load(args)
    grid_size = args.grid_size
    rollout_steps = _resolve_num_steps(args, model.timestep_jump, default_time_over_tau=50.0)
    T0 = _make_T0(grid_size, args.bc_left, args.bc_right)

    _warm_up(model, device, grid_size, args)

    q_t = torch.zeros(1, grid_size, device=device)
    tau_t = torch.full((1, grid_size), args.tau, device=device)
    alpha_t = torch.full((1, grid_size), args.alpha, device=device)
    rho_cp_t = torch.full((1, grid_size), args.rho_cp, device=device)
    bc_l = torch.tensor([args.bc_left], device=device)
    bc_r = torch.tensor([args.bc_right], device=device)
    # dt_eff = args.dt already includes timestep_jump
    dt_eff = args.dt
    dt_base = args.dt / max(1, model.timestep_jump)

    T_t = torch.from_numpy(T0).float().unsqueeze(0).to(device)
    T_prev_t = T_t.clone()
    neural_history = [T0.copy()]
    hidden_state = None

    _cuda_sync()
    t0 = time.perf_counter()
    neural_history_gpu = []
    with torch.inference_mode():
        for _ in range(rollout_steps):
            output = model(
                T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                bc_l, bc_r, dt_eff, args.dx,
                hidden_state=hidden_state,
            )
            T_next = output['T_pred']
            hidden_state = output.get('hidden_state')
            T_next[:, 0] = args.bc_left
            T_next[:, -1] = args.bc_right
            T_prev_t = T_t
            T_t = T_next
            neural_history_gpu.append(T_t.squeeze(0).detach())
    _cuda_sync()
    neural_time = time.perf_counter() - t0

    # Transfer all results to CPU at once (avoids per-step sync)
    neural_history = [T0.copy()] + [t.cpu().numpy() for t in neural_history_gpu]
    del neural_history_gpu

    total_fdm_equiv_steps = rollout_steps * model.timestep_jump
    fdm_history, fdm_time = _run_pure_fdm(
        T0, grid_size, args.dx, dt_base, args.tau,
        total_fdm_equiv_steps, args.bc_left, args.bc_right,
        save_every=max(1, model.timestep_jump),
    )

    n_compare = min(len(neural_history), len(fdm_history))
    neural_history = neural_history[:n_compare]
    fdm_history = fdm_history[:n_compare]

    neural_final = neural_history[-1]
    fdm_final = fdm_history[-1]
    rel_err = np.linalg.norm(neural_final - fdm_final) / (np.linalg.norm(fdm_final) + 1e-15) * 100
    max_err = np.max(np.abs(neural_final - fdm_final))

    step_errors = []
    for T_neural, T_fdm in zip(neural_history[1:], fdm_history[1:]):
        e = np.linalg.norm(T_neural - T_fdm) / (np.linalg.norm(T_fdm) + 1e-15) * 100
        step_errors.append(e)

    metrics = dict(
        mode='rollout',
        neural_time_s=neural_time,
        fdm_time_s=fdm_time,
        speedup=fdm_time / (neural_time + 1e-15),
        rel_error_pct=rel_err,
        max_abs_error=float(max_err),
        mean_step_error_pct=float(np.mean(step_errors)) if step_errors else 0.0,
        max_step_error_pct=float(np.max(step_errors)) if step_errors else 0.0,
        rollout_steps=rollout_steps,
        total_fdm_equiv_steps=total_fdm_equiv_steps,
        physical_time_s=rollout_steps * args.dt,  # rollout_steps * dt_stored = total physical time
        physical_time_over_tau=(rollout_steps * args.dt) / (args.tau + 1e-30),
    )
    _print_metrics(metrics)
    _mean_T_change(T0, fdm_final)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(grid_size)

    ax = axes[0]
    ax.plot(x, fdm_final, 'b-', lw=2, label='Pure FDM')
    ax.plot(x, neural_final, 'r--', lw=2, label='Neural rollout')
    ax.set_xlabel('Grid point')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Final Temperature Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, len(step_errors) + 1), step_errors, 'g-', lw=1.5)
    ax.set_xlabel('Rollout step')
    ax.set_ylabel('Relative error [%]')
    ax.set_title('Autoregressive Error vs FDM')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    n_show = min(n_compare, 8)
    idx = np.linspace(0, n_compare - 1, n_show, dtype=int)
    for i, step_idx in enumerate(idx):
        a = 0.3 + 0.7 * i / max(1, n_show)
        ax.plot(x, neural_history[step_idx], 'r-', alpha=a, lw=0.8)
        ax.plot(x, fdm_history[step_idx], 'b-', alpha=a, lw=0.8)
    ax.set_xlabel('Grid point')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Evolution (red=neural, blue=FDM)')
    ax.grid(True, alpha=0.3)

    _save_fig(fig, 'figures/benchmark_rollout.png')
    _save_json(metrics, 'results/benchmark_rollout.json')
    _save_gif(neural_history, list(fdm_history), x,
              'Rollout Temperature Evolution: Neural vs FDM',
              'figures/benchmark_rollout.gif',
              neural_label='Neural rollout', fdm_label='Pure FDM')

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 1: Hybrid solver (neural + FDM fallback) vs pure FDM
# ═══════════════════════════════════════════════════════════════════════

def benchmark_hybrid(args):
    """Run hybrid neural solver vs pure FDM over the same physical time."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Hybrid Solver (Neural + FDM fallback)  vs  Pure FDM")
    print("=" * 70)

    model, device = _load(args)
    grid_size = args.grid_size
    T0 = _make_T0(grid_size, args.bc_left, args.bc_right)
    grid = np.ones(grid_size, dtype=int)
    alpha = np.ones(grid_size) * args.alpha
    rho_cp = np.ones(grid_size) * args.rho_cp

    _warm_up(model, device, grid_size, args)

    # ── Neural hybrid ──
    solver_config = HybridSolverConfig(
        residual_threshold=2.0,
        gradient_threshold=500.0,
        timestep_jump=args.timestep_jump,
        max_timestep_jump=args.timestep_jump,
        verbose=False,
    )
    dt_base = args.dt / max(1, args.timestep_jump)
    solver = HybridCattaneoSolver(grid_size, args.dx, dt_base, model, solver_config)
    num_steps = _resolve_num_steps(args, args.timestep_jump, default_time_over_tau=25.0)

    _cuda_sync()
    t0 = time.perf_counter()
    neural_result = solver.solve(
        T0, grid, num_steps=num_steps, tau=args.tau,
        alpha=alpha, rho_cp=rho_cp,
        bc_left=args.bc_left, bc_right=args.bc_right,
        save_history=True,
    )
    _cuda_sync()
    neural_time = time.perf_counter() - t0

    # Compute how many FDM steps the neural solver covered
    total_fdm_equiv = neural_result.get('total_effective_fdm_steps',
                                         num_steps * args.timestep_jump)

    # ── Pure FDM over same physical time ──
    save_every_hybrid = max(1, total_fdm_equiv // max(1, num_steps))
    fdm_history, fdm_time = _run_pure_fdm(
        T0, grid_size, args.dx, dt_base, args.tau,
        total_fdm_equiv, args.bc_left, args.bc_right,
        save_every=save_every_hybrid,
    )

    # ── Metrics ──
    neural_final = neural_result['T_final']
    fdm_final = fdm_history[-1]
    rel_err = np.linalg.norm(neural_final - fdm_final) / (np.linalg.norm(fdm_final) + 1e-15) * 100
    max_err = np.max(np.abs(neural_final - fdm_final))
    neural_pct = neural_result.get('neural_percentage', 0)

    metrics = dict(
        mode='hybrid',
        neural_time_s=neural_time,
        fdm_time_s=fdm_time,
        speedup=fdm_time / (neural_time + 1e-15),
        rel_error_pct=rel_err,
        max_abs_error=float(max_err),
        neural_steps=neural_result['stats']['neural_calls'],
        fdm_steps_by_hybrid=neural_result['stats']['fdm_calls'],
        total_fdm_equiv_steps=total_fdm_equiv,
        physical_time_over_tau=(total_fdm_equiv * dt_base) / (args.tau + 1e-30),
        neural_pct=neural_pct,
    )
    _print_metrics(metrics)
    _mean_T_change(T0, fdm_final)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(grid_size)

    ax = axes[0]
    ax.plot(x, fdm_final, 'b-', lw=2, label='Pure FDM')
    ax.plot(x, neural_final, 'r--', lw=2, label='Hybrid (Neural+FDM)')
    ax.set_xlabel('Grid point')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Final Temperature Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(x, np.abs(neural_final - fdm_final) + 1e-15, 'g-', lw=2)
    ax.set_xlabel('Grid point')
    ax.set_ylabel('|T_neural − T_fdm| [K]')
    ax.set_title(f'Pointwise Error  (rel={rel_err:.4f}%)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    neural_hist = neural_result.get('T_history')
    if neural_hist is not None and len(neural_hist) > 1:
        step = max(1, len(neural_hist) // 8)
        for i in range(0, len(neural_hist), step):
            a = 0.3 + 0.7 * i / len(neural_hist)
            ax.plot(x, neural_hist[i], 'r-', alpha=a, lw=0.8)
        for i in range(0, len(fdm_history), max(1, len(fdm_history) // 8)):
            a = 0.3 + 0.7 * i / len(fdm_history)
            ax.plot(x, fdm_history[i], 'b-', alpha=a, lw=0.8)
    ax.set_xlabel('Grid point')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Temporal Evolution (red=neural, blue=FDM)')
    ax.grid(True, alpha=0.3)

    _save_fig(fig, 'figures/benchmark_hybrid.png')
    _save_json(metrics, 'results/benchmark_hybrid.json')

    # ── GIF ──
    neural_hist = neural_result.get('T_history')
    if neural_hist is not None and len(neural_hist) > 1:
        # Subsample FDM history to match neural history length
        n = len(neural_hist)
        fdm_idx = np.linspace(0, len(fdm_history) - 1, n, dtype=int)
        _save_gif(list(neural_hist), [fdm_history[i] for i in fdm_idx],
                  x, 'Hybrid Solver vs Pure FDM', 'figures/benchmark_hybrid.gif',
                  neural_label='Hybrid (Neural+FDM)', fdm_label='Pure FDM')

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 2: Super-resolution
# ═══════════════════════════════════════════════════════════════════════

def benchmark_super_res(args):
    """
    Compare super-resolution: LNO on fine grid vs pure FDM on fine grid.

    1. Start with coarse T0 (grid_size).
    2. Neural: interpolate to fine grid → LNO forward pass  → repeat num_steps.
    3. FDM:    interpolate to fine grid → FDM fine steps    → same physical time.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Super-Resolution  (LNO on fine grid  vs  FDM on fine grid)")
    print("=" * 70)

    model, device = _load(args)
    grid_size = args.grid_size
    fine_grid = grid_size * 2
    rollout_steps = min(_resolve_num_steps(args, model.timestep_jump, default_time_over_tau=15.0), 2500)
    T0_coarse = _make_T0(grid_size, args.bc_left, args.bc_right)

    _warm_up(model, device, fine_grid, args)

    # ── Neural super-resolution rollout ──
    sr = SuperResolutionInference(model, device=device)
    _cuda_sync()
    t0 = time.perf_counter()
    sr_history, dt_eff_fine = sr.rollout(
        T0_coarse, fine_grid, num_steps=rollout_steps,
        tau=args.tau, alpha=args.alpha, rho_cp=args.rho_cp,
        bc_left=args.bc_left, bc_right=args.bc_right,
        dt=args.dt, dx=args.dx,
    )
    _cuda_sync()
    neural_time = time.perf_counter() - t0

    # ── Pure FDM on the fine grid for same physical time ──
    from scipy.interpolate import interp1d
    x_coarse = np.linspace(0, 1, grid_size)
    x_fine = np.linspace(0, 1, fine_grid)
    T0_fine = interp1d(x_coarse, T0_coarse, kind='cubic',
                       fill_value='extrapolate')(x_fine)

    # dt_eff_fine is scaled to conserve Fo.  Total physical time
    # covered = rollout_steps * dt_eff_fine.  FDM uses base dt with
    # fine dx → total_fdm_fine_steps = total_physical_time / dt.
    L = grid_size * args.dx
    dx_fine = L / fine_grid
    dt_base = args.dt / max(1, model.timestep_jump)
    total_physical_time = rollout_steps * dt_eff_fine
    total_fdm_fine_steps = int(round(total_physical_time / dt_base))
    # Save every N steps so history length roughly matches neural
    save_every_fine = max(1, total_fdm_fine_steps // rollout_steps)

    fdm_history, fdm_time = _run_pure_fdm(
        T0_fine, fine_grid, dx_fine, dt_base, args.tau,
        total_fdm_fine_steps, args.bc_left, args.bc_right,
        save_every=save_every_fine,
    )

    # ── Metrics ──
    neural_final = sr_history[-1]
    fdm_final = fdm_history[-1]
    rel_err = np.linalg.norm(neural_final - fdm_final) / (np.linalg.norm(fdm_final) + 1e-15) * 100
    max_err = np.max(np.abs(neural_final - fdm_final))

    metrics = dict(
        mode='super-res',
        coarse_grid=grid_size,
        fine_grid=fine_grid,
        rollout_steps=rollout_steps,
        neural_time_s=neural_time,
        fdm_time_s=fdm_time,
        speedup=fdm_time / (neural_time + 1e-15),
        rel_error_pct=rel_err,
        max_abs_error=float(max_err),
        physical_time_over_tau=total_physical_time / (args.tau + 1e-30),
    )
    _print_metrics(metrics)
    _mean_T_change(T0_fine, fdm_final)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(fine_grid)

    ax = axes[0]
    ax.plot(x, fdm_final, 'b-', lw=2, label=f'FDM (fine {fine_grid})')
    ax.plot(x, neural_final, 'r--', lw=2, label=f'Neural SR (fine {fine_grid})')
    ax.set_xlabel('Grid point (fine)')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Super-Resolution: Final Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(x, np.abs(neural_final - fdm_final) + 1e-15, 'g-', lw=2)
    ax.set_xlabel('Grid point (fine)')
    ax.set_ylabel('|T_neural − T_fdm| [K]')
    ax.set_title(f'Pointwise Error  (rel={rel_err:.4f}%)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    n_show = min(len(sr_history), len(fdm_history), 8)
    sr_idx = np.linspace(0, len(sr_history) - 1, n_show, dtype=int)
    fdm_idx = np.linspace(0, len(fdm_history) - 1, n_show, dtype=int)
    for i, (si, fi) in enumerate(zip(sr_idx, fdm_idx)):
        a = 0.3 + 0.7 * i / n_show
        ax.plot(x, sr_history[si], 'r-', alpha=a, lw=0.8)
        ax.plot(x, fdm_history[fi], 'b-', alpha=a, lw=0.8)
    ax.set_xlabel('Grid point (fine)')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Evolution (red=neural, blue=FDM)')
    ax.grid(True, alpha=0.3)

    _save_fig(fig, 'figures/benchmark_super_res.png')
    _save_json(metrics, 'results/benchmark_super_res.json')

    # ── GIF ──
    n = min(len(sr_history), len(fdm_history))
    fdm_sub = [fdm_history[i] for i in np.linspace(0, len(fdm_history) - 1, n, dtype=int)]
    _save_gif(sr_history[:n], fdm_sub, x,
              'Super-Resolution: Neural vs FDM (fine grid)',
              'figures/benchmark_super_res.gif',
              xlabel='Grid point (fine)',
              neural_label='Neural SR', fdm_label='FDM (fine)')

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 3: Temporal Frame Generation
# ═══════════════════════════════════════════════════════════════════════

def benchmark_frame_gen(args):
    """
    Compare dense temporal frame generation:
      Neural: LNO sub-stepping + linear interpolation
      FDM:    raw FDM at every dt, sampled at same output times
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Frame Generation  (Neural sub-stepping  vs  FDM)")
    print("=" * 70)

    model, device = _load(args)
    grid_size = args.grid_size
    T0 = _make_T0(grid_size, args.bc_left, args.bc_right)

    num_frames = _resolve_num_steps(args, model.timestep_jump, default_time_over_tau=10.0)
    sub_steps = 1

    _warm_up(model, device, grid_size, args)

    # ── Neural frame generation ──
    fg = TemporalFrameGenerator(model, device=device)
    _cuda_sync()
    t0 = time.perf_counter()
    neural_frames = fg.generate(
        T0, num_frames=num_frames, sub_steps=sub_steps,
        tau=args.tau, alpha=args.alpha, rho_cp=args.rho_cp,
        bc_left=args.bc_left, bc_right=args.bc_right,
        dt=args.dt, dx=args.dx,
    )
    _cuda_sync()
    neural_time = time.perf_counter() - t0

    # ── Pure FDM producing same number of output frames ──
    # The neural frame generator takes ceil(num_frames/sub_steps) neural steps.
    # Each neural step = timestep_jump FDM steps.
    # Total physical FDM steps = neural_steps * timestep_jump
    dt_base = args.dt / max(1, model.timestep_jump)
    neural_steps_taken = (num_frames + sub_steps - 1) // sub_steps
    total_fdm_steps = neural_steps_taken * model.timestep_jump
    # Sample FDM every (total_fdm_steps / num_frames) steps
    save_every = max(1, total_fdm_steps // num_frames)

    fdm_history, fdm_time = _run_pure_fdm(
        T0, grid_size, args.dx, dt_base, args.tau,
        total_fdm_steps, args.bc_left, args.bc_right,
        save_every=save_every,
    )

    # Align frame counts
    n_compare = min(len(neural_frames), len(fdm_history))
    neural_frames = neural_frames[:n_compare]
    fdm_frames = fdm_history[:n_compare]

    # ── Metrics ──
    neural_final = neural_frames[-1]
    fdm_final = fdm_frames[-1]
    rel_err = np.linalg.norm(neural_final - fdm_final) / (np.linalg.norm(fdm_final) + 1e-15) * 100
    max_err = np.max(np.abs(neural_final - fdm_final))

    # Frame-by-frame errors
    frame_errors = []
    for nf, ff in zip(neural_frames, fdm_frames):
        e = np.linalg.norm(nf - ff) / (np.linalg.norm(ff) + 1e-15) * 100
        frame_errors.append(e)

    metrics = dict(
        mode='frame-gen',
        num_frames=num_frames,
        sub_steps=sub_steps,
        neural_time_s=neural_time,
        fdm_time_s=fdm_time,
        speedup=fdm_time / (neural_time + 1e-15),
        rel_error_final_pct=rel_err,
        max_abs_error=float(max_err),
        mean_frame_error_pct=float(np.mean(frame_errors)),
        max_frame_error_pct=float(np.max(frame_errors)),
        physical_time_over_tau=(total_fdm_steps * args.dt) / (args.tau + 1e-30),
    )
    _print_metrics(metrics)
    _mean_T_change(T0, fdm_final)

    # ── Plot ──
    x = np.arange(grid_size)
    if not getattr(args, 'skip_figures', False):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        ax.plot(x, fdm_final, 'b-', lw=2, label='FDM')
        ax.plot(x, neural_final, 'r--', lw=2, label='Neural Frames')
        ax.set_xlabel('Grid point')
        ax.set_ylabel('Temperature [K]')
        ax.set_title('Final Frame Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(range(len(frame_errors)), frame_errors, 'g-o', ms=3, lw=1.5)
        ax.set_xlabel('Frame index')
        ax.set_ylabel('Relative error [%]')
        ax.set_title('Per-Frame Error vs FDM')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        n_show = min(n_compare, 8)
        idx = np.linspace(0, n_compare - 1, n_show, dtype=int)
        for i, fi in enumerate(idx):
            a = 0.3 + 0.7 * i / n_show
            ax.plot(x, neural_frames[fi], 'r-', alpha=a, lw=0.8)
            ax.plot(x, fdm_frames[fi], 'b-', alpha=a, lw=0.8)
        ax.set_xlabel('Grid point')
        ax.set_ylabel('Temperature [K]')
        ax.set_title('Evolution (red=neural, blue=FDM)')
        ax.grid(True, alpha=0.3)

        _save_fig(fig, 'figures/benchmark_frame_gen.png')
    if not getattr(args, 'skip_gif', False):
        _save_gif(neural_frames, list(fdm_frames),
                  x, 'Frame Generation: Neural vs FDM',
                  'figures/benchmark_frame_gen.gif',
                  neural_label='Neural Frames', fdm_label='FDM')

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 4: Neural Warm-Start
# ═══════════════════════════════════════════════════════════════════════

def benchmark_warm_start(args):
    """
    Compare warm-start (neural fast-forward + FDM refinement) vs
    cold-start FDM (pure FDM from t=0 for same total physical time).
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Warm-Start  (Neural + FDM refinement  vs  Cold-start FDM)")
    print("=" * 70)

    model, device = _load(args)
    grid_size = args.grid_size
    T0 = _make_T0(grid_size, args.bc_left, args.bc_right)
    grid = np.ones(grid_size, dtype=int)

    num_neural_steps = min(_resolve_num_steps(args, model.timestep_jump, default_time_over_tau=25.0), 2500)
    num_fdm_refine = 500

    _warm_up(model, device, grid_size, args)

    # ── Neural warm-start + FDM refinement ──
    ws = NeuralWarmStart(model, device=device)
    _cuda_sync()
    t0 = time.perf_counter()
    ws_result = ws.solve(
        T0, grid,
        num_neural_steps=num_neural_steps,
        num_fdm_steps=num_fdm_refine,
        tau=args.tau, alpha=args.alpha, rho_cp=args.rho_cp,
        bc_left=args.bc_left, bc_right=args.bc_right,
        dx=args.dx, dt=args.dt,
    )
    _cuda_sync()
    warm_time = time.perf_counter() - t0
    T_warm = ws_result['T_final']

    # Total FDM-equivalent steps for warm-start:
    #   neural phase: num_neural_steps * timestep_jump FDM sub-steps
    #   FDM phase:    num_fdm_refine * timestep_jump FDM sub-steps
    total_warm_equiv = (num_neural_steps + num_fdm_refine) * args.timestep_jump

    # ── Pure FDM cold-start for same total physical time ──
    save_every_ws = max(1, total_warm_equiv // (num_neural_steps + num_fdm_refine))
    fdm_history, fdm_time = _run_pure_fdm(
        T0, grid_size, args.dx, args.dt, args.tau,
        total_warm_equiv, args.bc_left, args.bc_right,
        save_every=save_every_ws,
    )

    # ── Also run pure FDM for only the FDM-refinement phase
    #    (starting from same point the neural reached) ──
    #    This shows how much faster the neural phase was vs FDM
    neural_fdm_equiv_steps = num_neural_steps * args.timestep_jump
    save_every_np = max(1, neural_fdm_equiv_steps // num_neural_steps)
    fdm_neural_phase_hist, fdm_neural_phase_time = _run_pure_fdm(
        T0, grid_size, args.dx, args.dt, args.tau,
        neural_fdm_equiv_steps, args.bc_left, args.bc_right,
        save_every=save_every_np,
    )
    T_fdm_at_switchover = fdm_neural_phase_hist[-1]
    print(f"    Pure FDM at switchover: T range [{T_fdm_at_switchover.min():.2f}, {T_fdm_at_switchover.max():.2f}]")

    # ── Metrics ──
    fdm_final = fdm_history[-1]
    rel_err = np.linalg.norm(T_warm - fdm_final) / (np.linalg.norm(fdm_final) + 1e-15) * 100
    max_err = np.max(np.abs(T_warm - fdm_final))

    metrics = dict(
        mode='warm-start',
        num_neural_steps=num_neural_steps,
        num_fdm_refine=num_fdm_refine,
        total_fdm_equiv_steps=total_warm_equiv,
        warm_start_time_s=warm_time,
        cold_start_fdm_time_s=fdm_time,
        speedup=fdm_time / (warm_time + 1e-15),
        rel_error_pct=rel_err,
        max_abs_error=float(max_err),
        fdm_neural_phase_time_s=fdm_neural_phase_time,
        physical_time_over_tau=(total_warm_equiv * args.dt) / (args.tau + 1e-30),
    )
    _print_metrics(metrics)
    _mean_T_change(T0, fdm_final)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(grid_size)

    ax = axes[0]
    ax.plot(x, fdm_final, 'b-', lw=2, label='Cold-start FDM')
    ax.plot(x, T_warm, 'r--', lw=2, label='Warm-start (Neural+FDM)')
    ax.plot(x, T_fdm_at_switchover, 'k:', lw=1.5, label='FDM at switchover')
    ax.set_xlabel('Grid point')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Final Temperature Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(x, np.abs(T_warm - fdm_final) + 1e-15, 'g-', lw=2)
    ax.set_xlabel('Grid point')
    ax.set_ylabel('|T_warm − T_cold| [K]')
    ax.set_title(f'Error vs Cold-Start  (rel={rel_err:.4f}%)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = ['Cold-start\nFDM', 'Warm-start\n(Neural+FDM)']
    times = [fdm_time, warm_time]
    colors = ['steelblue', 'tomato']
    bars = ax.bar(labels, times, color=colors, width=0.5)
    ax.set_ylabel('Wall-clock time [s]')
    ax.set_title(f'Timing  (speedup={fdm_time / (warm_time + 1e-15):.2f}×)')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{t:.3f}s', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    _save_fig(fig, 'figures/benchmark_warm_start.png')
    _save_json(metrics, 'results/benchmark_warm_start.json')

    # ── GIF: compare cold-start FDM evolution with switchover point ──
    # Use the FDM history (saved every step) vs a simple 3-frame animation
    # showing initial → switchover → final for warm-start
    n_fdm = len(fdm_history)
    ws_frames = [T0.copy(), T_fdm_at_switchover.copy(), T_warm.copy()]
    # Map warm-start 3 key frames to corresponding FDM frames
    fdm_key_idx = [0, len(fdm_neural_phase_hist) - 1, n_fdm - 1]
    fdm_key_frames = [fdm_history[i] for i in fdm_key_idx]
    _save_gif(ws_frames, fdm_key_frames, x,
              'Warm-Start vs Cold-Start FDM',
              'figures/benchmark_warm_start.gif',
              neural_label='Warm-start', fdm_label='Cold-start FDM', fps=2)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 5: System-size scaling
# ═══════════════════════════════════════════════════════════════════════

def benchmark_scaling(args):
    """Measure how neural and FDM wall-clock time scale with system size.

    Strategy: keep dx = dx_train fixed (so Fo is the same as training),
    and increase N to make the physical domain larger.  This is the
    standard "weak scaling" test — bigger system, same resolution.

    For each grid size N:
      - Neural: N steps of LNO forward pass (same Fo, same dt, same dx)
      - FDM:    N_fdm = neural_steps × timestep_jump steps of implicit
                SuperLU back-substitution

    Expected scaling:
      - FDM (SuperLU pre-factored): O(N) per step, O(N² · steps) total
        (factorisation is O(N) for tridiag, solve is O(N) per step)
      - Neural (LNO): O(N·modes) per step — spectral conv + point-wise heads
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: System-Size Scaling  (Neural vs FDM)")
    print("=" * 70)

    model, device = _load(args)

    dx = args.dx              # fixed at training value
    dt_stored = args.dt       # effective dt (includes timestep_jump)
    dt_base = dt_stored / max(1, model.timestep_jump)
    tj = model.timestep_jump
    Fo_eff = args.alpha * dt_stored / (dx ** 2)

    # Grid sizes to test
    grid_sizes = [56, 112, 224, 448, 896, 1792]

    time_over_tau = 5.0
    total_physical_time = time_over_tau * args.tau

    neural_steps = max(1, int(np.ceil(total_physical_time / dt_stored)))
    fdm_total_steps = neural_steps * tj

    results = []
    print(f"\n  dx = {dx:.2e} m (fixed),  Fo_eff = {Fo_eff:.3f} (constant)")
    print(f"  physical time = {time_over_tau:.1f}τ = {total_physical_time:.2e} s")
    print(f"  Neural steps = {neural_steps},  FDM steps = {fdm_total_steps}")
    print()
    header = (f"  {'N':>6}  {'L [m]':>10}  "
              f"{'Neural [s]':>12}  {'FDM [s]':>12}  {'Speedup':>9}  "
              f"{'Rel Err [%]':>13}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for gs in grid_sizes:
        L_gs = gs * dx     # physical domain length for this N

        T0 = _make_T0(gs, args.bc_left, args.bc_right)

        # ── Neural rollout ──
        q_t = torch.zeros(1, gs, device=device)
        tau_t = torch.full((1, gs), args.tau, device=device)
        alpha_t = torch.full((1, gs), args.alpha, device=device)
        rho_cp_t = torch.full((1, gs), args.rho_cp, device=device)
        bc_l = torch.tensor([args.bc_left], device=device)
        bc_r = torch.tensor([args.bc_right], device=device)

        T_t = torch.from_numpy(T0).float().unsqueeze(0).to(device)
        T_prev_t = T_t.clone()
        hidden_state = None

        # Warmup (3 passes to trigger compilation / cache for this size)
        with torch.inference_mode():
            for _ in range(3):
                model(T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                      bc_l, bc_r, dt_stored, dx, hidden_state=hidden_state)
        _cuda_sync()

        # Timed rollout
        T_t = torch.from_numpy(T0).float().unsqueeze(0).to(device)
        T_prev_t = T_t.clone()
        hidden_state = None
        _cuda_sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(neural_steps):
                out = model(
                    T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                    bc_l, bc_r, dt_stored, dx,
                    hidden_state=hidden_state,
                )
                T_prev_t = T_t
                T_t = out['T_pred']
                T_t[:, 0] = args.bc_left
                T_t[:, -1] = args.bc_right
                hidden_state = out.get('hidden_state')
        _cuda_sync()
        neural_time = time.perf_counter() - t0
        neural_final = T_t.squeeze(0).cpu().numpy()

        # ── FDM at this grid size (same dx, same dt_base) ──
        fdm_history, fdm_time = _run_pure_fdm(
            T0, gs, dx, dt_base, args.tau,
            fdm_total_steps, args.bc_left, args.bc_right,
            save_every=fdm_total_steps,  # only keep final state
        )
        fdm_final = fdm_history[-1]

        rel_err = (np.linalg.norm(neural_final - fdm_final)
                   / (np.linalg.norm(fdm_final) + 1e-15) * 100)
        speedup = fdm_time / (neural_time + 1e-15)

        results.append(dict(
            grid_size=gs,
            domain_length_m=L_gs,
            Fo_eff=Fo_eff,
            neural_time_s=neural_time,
            fdm_time_s=fdm_time,
            speedup=speedup,
            rel_error_pct=rel_err,
            neural_steps=neural_steps,
            fdm_steps=fdm_total_steps,
        ))

        print(f"  {gs:>6}  {L_gs:>10.2e}  "
              f"{neural_time:>12.4f}  {fdm_time:>12.4f}  {speedup:>8.2f}x  "
              f"{rel_err:>13.4f}")

    metrics = dict(
        mode='scaling',
        dx=dx,
        Fo_eff=Fo_eff,
        time_over_tau=time_over_tau,
        physical_time_s=total_physical_time,
        neural_steps=neural_steps,
        fdm_total_steps=fdm_total_steps,
        dt_stored=dt_stored,
        dt_base=dt_base,
        timestep_jump=tj,
        results=results,
    )
    _save_json(metrics, 'results/benchmark_scaling.json')

    # ── Figure: log-log scaling plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sizes = [r['grid_size'] for r in results]
    t_nn = [r['neural_time_s'] for r in results]
    t_fdm = [r['fdm_time_s'] for r in results]
    speedups = [r['speedup'] for r in results]
    errors = [r['rel_error_pct'] for r in results]

    # Panel 1: wall-clock time
    ax = axes[0]
    ax.loglog(sizes, t_fdm, 'bs-', lw=2, ms=8, label='FDM (SuperLU)')
    ax.loglog(sizes, t_nn, 'ro-', lw=2, ms=8, label='Neural (LNO)')
    ax.set_xlabel('Grid points N')
    ax.set_ylabel('Wall-clock time [s]')
    ax.set_title(f'Timing ({neural_steps} steps, {time_over_tau:.0f}τ)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Reference slopes
    n_arr = np.array(sizes, dtype=float)
    ref_n1 = t_fdm[0] * (n_arr / n_arr[0])
    ax.loglog(n_arr, ref_n1, 'b:', alpha=0.4, label='O(N)')
    ref_nlogn = t_nn[0] * (n_arr * np.log2(n_arr)) / (n_arr[0] * np.log2(n_arr[0]))
    ax.loglog(n_arr, ref_nlogn, 'r:', alpha=0.4, label='O(N log N)')
    ax.legend(fontsize=9)

    # Panel 2: speedup
    ax = axes[1]
    ax.semilogx(sizes, speedups, 'g^-', lw=2, ms=8)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Grid points N')
    ax.set_ylabel('Speedup (FDM / Neural)')
    ax.set_title('Speedup vs System Size')
    ax.grid(True, alpha=0.3, which='both')
    ax.fill_between(sizes, 1.0, [max(s, 1.0) for s in speedups],
                     alpha=0.15, color='green',
                     where=[s > 1.0 for s in speedups])
    ax.fill_between(sizes, [min(s, 1.0) for s in speedups], 1.0,
                     alpha=0.15, color='red',
                     where=[s < 1.0 for s in speedups])

    # Panel 3: accuracy
    ax = axes[2]
    ax.semilogx(sizes, errors, 'mp-', lw=2, ms=8)
    ax.set_xlabel('Grid points N')
    ax.set_ylabel('Relative error vs FDM [%]')
    ax.set_title('Accuracy vs System Size')
    ax.grid(True, alpha=0.3, which='both')

    _save_fig(fig, 'figures/benchmark_scaling.png')
    _print_metrics({k: v for k, v in metrics.items() if k != 'results'})

    # Print crossover point
    for r in results:
        if r['speedup'] >= 1.0:
            print(f"  *** Neural breaks even at N={r['grid_size']} "
                  f"(speedup={r['speedup']:.2f}x) ***")
            break

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Benchmark 6: Temperature evolution at multiple resolutions (fixed L)
# ═══════════════════════════════════════════════════════════════════════

def benchmark_scaling_evolution(args):
    """Show temperature evolution at different grid resolutions for a fixed domain.

    Keeps the physical domain length L = N_train × dx_train fixed.
    For each resolution N, adjusts dt so that Fo_eff matches training,
    keeping the neural model in its stable operating regime.

    Produces a multi-panel figure: one row per grid size, showing
    snapshots of T(x) at matched physical times for both neural and FDM.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Resolution Convergence  (fixed domain, varying N)")
    print("=" * 70)

    model, device = _load(args)

    gs_train = args.grid_size
    dx_train = args.dx
    L = gs_train * dx_train  # fixed domain length

    dt_stored_train = args.dt
    tj = model.timestep_jump
    Fo_train = args.alpha * dt_stored_train / (dx_train ** 2)

    time_over_tau = 10.0
    total_physical_time = time_over_tau * args.tau

    grid_sizes = [112, 224, 448, 896]
    # Snapshot times (fractions of total time) for the evolution panels
    snap_fracs = [0.0, 0.1, 0.25, 0.5, 1.0]

    print(f"\n  Domain L = {L:.2e} m,  Fo_train = {Fo_train:.3f}")
    print(f"  physical time = {time_over_tau:.0f}τ = {total_physical_time:.2e} s")
    print(f"  Resolutions: {grid_sizes}")
    print()

    all_data = {}  # grid_size -> dict of results

    for gs in grid_sizes:
        dx_gs = L / gs
        # Scale dt so that Fo stays the same as training
        dt_eff = Fo_train * (dx_gs ** 2) / args.alpha
        dt_base_gs = dt_eff / tj

        neural_steps = max(1, int(np.ceil(total_physical_time / dt_eff)))
        fdm_total = neural_steps * tj

        # Determine which neural steps correspond to each snapshot
        snap_steps = [max(0, int(round(f * neural_steps))) for f in snap_fracs]
        snap_steps[-1] = neural_steps  # ensure last is exact
        # Corresponding FDM save indices
        fdm_save_every = max(1, tj)

        print(f"  N={gs:>5}  dx={dx_gs:.2e}  dt_eff={dt_eff:.2e}  "
              f"Fo={args.alpha * dt_eff / dx_gs**2:.3f}  "
              f"neural_steps={neural_steps}  fdm_steps={fdm_total}")

        T0 = _make_T0(gs, args.bc_left, args.bc_right)
        x_phys = np.linspace(0, L * 1e6, gs)  # in μm for plotting

        # ── Neural rollout with snapshot capture ──
        q_t = torch.zeros(1, gs, device=device)
        tau_t = torch.full((1, gs), args.tau, device=device)
        alpha_t = torch.full((1, gs), args.alpha, device=device)
        rho_cp_t = torch.full((1, gs), args.rho_cp, device=device)
        bc_l = torch.tensor([args.bc_left], device=device)
        bc_r = torch.tensor([args.bc_right], device=device)

        # Warmup
        T_t = torch.from_numpy(T0).float().unsqueeze(0).to(device)
        T_prev_t = T_t.clone()
        with torch.inference_mode():
            for _ in range(3):
                model(T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                      bc_l, bc_r, dt_eff, dx_gs)
        _cuda_sync()

        # Rollout
        T_t = torch.from_numpy(T0).float().unsqueeze(0).to(device)
        T_prev_t = T_t.clone()
        hidden_state = None
        snap_set = set(snap_steps)
        neural_snaps = {0: T0.copy()}

        _cuda_sync()
        t0_wall = time.perf_counter()
        with torch.inference_mode():
            for step in range(1, neural_steps + 1):
                out = model(
                    T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                    bc_l, bc_r, dt_eff, dx_gs,
                    hidden_state=hidden_state,
                )
                T_prev_t = T_t
                T_t = out['T_pred']
                T_t[:, 0] = args.bc_left
                T_t[:, -1] = args.bc_right
                hidden_state = out.get('hidden_state')
                if step in snap_set:
                    neural_snaps[step] = T_t.squeeze(0).cpu().numpy()
        _cuda_sync()
        neural_time = time.perf_counter() - t0_wall

        # ── FDM rollout ──
        fdm_history, fdm_time = _run_pure_fdm(
            T0, gs, dx_gs, dt_base_gs, args.tau,
            fdm_total, args.bc_left, args.bc_right,
            save_every=tj,  # save every TJ steps = 1 neural step equivalent
        )
        # Build FDM snapshots at matching neural steps
        fdm_snaps = {}
        for s in snap_steps:
            fdm_idx = min(s, len(fdm_history) - 1)
            fdm_snaps[s] = fdm_history[fdm_idx]

        neural_final = neural_snaps[neural_steps]
        fdm_final = fdm_snaps[neural_steps]
        rel_err = (np.linalg.norm(neural_final - fdm_final)
                   / (np.linalg.norm(fdm_final) + 1e-15) * 100)

        all_data[gs] = dict(
            x_phys=x_phys,
            dx=dx_gs,
            dt_eff=dt_eff,
            neural_steps=neural_steps,
            fdm_steps=fdm_total,
            snap_steps=snap_steps,
            neural_snaps=neural_snaps,
            fdm_snaps=fdm_snaps,
            neural_time=neural_time,
            fdm_time=fdm_time,
            rel_err=rel_err,
        )
        print(f"          neural={neural_time:.2f}s  fdm={fdm_time:.2f}s  "
              f"rel_err={rel_err:.4f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Figure: multi-panel evolution comparison
    # ═══════════════════════════════════════════════════════════════════
    n_rows = len(grid_sizes)
    n_snap = len(snap_fracs)
    fig, axes = plt.subplots(n_rows, n_snap, figsize=(4 * n_snap, 3.5 * n_rows),
                             squeeze=False)

    for row, gs in enumerate(grid_sizes):
        d = all_data[gs]
        x = d['x_phys']

        for col, (frac, step) in enumerate(zip(snap_fracs, d['snap_steps'])):
            ax = axes[row, col]
            T_fdm = d['fdm_snaps'][step]
            T_nn = d['neural_snaps'][step]

            ax.plot(x, T_fdm, 'b-', lw=1.8, label='FDM')
            ax.plot(x, T_nn, 'r--', lw=1.5, label='Neural')

            phys_t = frac * total_physical_time
            t_over_tau = phys_t / args.tau

            if row == 0:
                ax.set_title(f't = {t_over_tau:.1f}τ', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'N={gs}\nT [K]', fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel('x [μm]')
            if row == 0 and col == n_snap - 1:
                ax.legend(fontsize=8, loc='upper right')

            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)

            # Show error in corner for non-initial snapshots
            if step > 0:
                err = (np.linalg.norm(T_nn - T_fdm)
                       / (np.linalg.norm(T_fdm) + 1e-15) * 100)
                ax.text(0.02, 0.02, f'{err:.3f}%', transform=ax.transAxes,
                        fontsize=7, color='gray', va='bottom')

    fig.suptitle(
        f'Temperature Evolution: Neural vs FDM at Different Resolutions\n'
        f'(fixed domain L = {L*1e6:.2f} μm,  {time_over_tau:.0f}τ,  '
        f'Fo = {Fo_train:.2f},  TJ = {tj})',
        fontsize=13, fontweight='bold', y=1.01)

    _save_fig(fig, 'figures/benchmark_scaling_evolution.png')

    # ── Summary table ──
    print()
    header = (f"  {'N':>6}  {'dx [m]':>10}  {'dt_eff':>10}  "
              f"{'NN steps':>10}  {'FDM steps':>10}  "
              f"{'NN [s]':>8}  {'FDM [s]':>8}  {'Err [%]':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for gs in grid_sizes:
        d = all_data[gs]
        print(f"  {gs:>6}  {d['dx']:>10.2e}  {d['dt_eff']:>10.2e}  "
              f"{d['neural_steps']:>10}  {d['fdm_steps']:>10}  "
              f"{d['neural_time']:>8.2f}  {d['fdm_time']:>8.2f}  "
              f"{d['rel_err']:>9.4f}")

    # ── Error convergence figure ──
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    gs_arr = np.array(grid_sizes, dtype=float)
    errs = [all_data[gs]['rel_err'] for gs in grid_sizes]
    ax2.loglog(gs_arr, errs, 'ro-', lw=2, ms=8, label='Neural vs FDM error')
    # Reference: 2nd-order convergence O(h²) ~ O(1/N²)
    if len(errs) >= 2 and errs[0] > 0:
        ref = errs[0] * (gs_arr[0] / gs_arr) ** 2
        ax2.loglog(gs_arr, ref, 'k:', alpha=0.5, label='O(1/N²) reference')
    ax2.set_xlabel('Grid points N')
    ax2.set_ylabel('Relative error vs FDM [%]')
    ax2.set_title('Resolution Convergence (fixed domain)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    _save_fig(fig2, 'figures/benchmark_scaling_convergence.png')

    metrics = dict(
        mode='scaling-evolution',
        domain_length_m=L,
        Fo_train=Fo_train,
        time_over_tau=time_over_tau,
        timestep_jump=tj,
        grid_sizes=grid_sizes,
        results={gs: dict(dx=all_data[gs]['dx'],
                          dt_eff=all_data[gs]['dt_eff'],
                          neural_steps=all_data[gs]['neural_steps'],
                          fdm_steps=all_data[gs]['fdm_steps'],
                          neural_time_s=all_data[gs]['neural_time'],
                          fdm_time_s=all_data[gs]['fdm_time'],
                          rel_error_pct=all_data[gs]['rel_err'])
                 for gs in grid_sizes},
    )
    _save_json(metrics, 'results/benchmark_scaling_evolution.json')

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

def run_all(args):
    """Run all benchmarks and print a summary table."""
    all_metrics = {}
    for mode_fn, name in [(benchmark_rollout, 'rollout'),
                          (benchmark_hybrid, 'hybrid'),
                          (benchmark_super_res, 'super-res'),
                          (benchmark_frame_gen, 'frame-gen')]:
        try:
            m = mode_fn(args)
            all_metrics[name] = m
        except Exception as e:
            import traceback
            print(f"\n  WARNING: {name} benchmark failed: {e}")
            traceback.print_exc()
            all_metrics[name] = {'error': str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Mode':<16} {'Neural [s]':>12} {'FDM [s]':>12} {'Speedup':>10} {'Rel Err [%]':>14}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        if 'error' in m:
            print(f"{name:<16} {'FAILED':>12}")
            continue
        nt = m.get('neural_time_s', m.get('warm_start_time_s', 0))
        ft = m.get('fdm_time_s', m.get('cold_start_fdm_time_s', 0))
        sp = m.get('speedup', 0)
        re = m.get('rel_error_pct', m.get('rel_error_final_pct', 0))
        print(f"{name:<16} {nt:>12.4f} {ft:>12.4f} {sp:>9.2f}x {re:>13.4f}")

    _save_json(all_metrics, 'results/benchmark_summary.json')
    return all_metrics


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def _mean_T_change(T0, T_final):
    """Compute and return the mean temperature change metrics."""
    mean_T0 = np.mean(T0)
    mean_Tf = np.mean(T_final)
    abs_change = abs(mean_Tf - mean_T0)
    pct_change = abs_change / (abs(mean_T0) + 1e-15) * 100
    print(f"  FDM mean T change: {mean_T0:.2f}K -> {mean_Tf:.2f}K "
          f"(delta={mean_Tf - mean_T0:+.2f}K, {pct_change:.1f}%)")
    return pct_change


def _print_metrics(m):
    """Pretty-print a metrics dict."""
    print()
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<30s} {v:.6g}")
        else:
            print(f"  {k:<30s} {v}")
    print()


def _save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure -> {path}")


def _save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)
    print(f"  Saved results -> {path}")


def _save_gif(neural_history, fdm_history, x, title, path,
              xlabel='Grid point', ylabel='Temperature [K]',
              neural_label='Neural', fdm_label='FDM', fps=10,
              max_gif_frames=80):
    """Create an animated GIF comparing neural vs FDM evolution.

    If there are more frames than *max_gif_frames*, subsample uniformly
    so each GIF frame represents a larger physical time-step.
    """
    n_raw = min(len(neural_history), len(fdm_history))

    # Subsample to keep GIF concise but each frame showing big jumps
    if n_raw > max_gif_frames:
        idx = np.linspace(0, n_raw - 1, max_gif_frames, dtype=int)
        neural_history = [neural_history[i] for i in idx]
        fdm_history = [fdm_history[i] for i in idx]

    n_frames = min(len(neural_history), len(fdm_history))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute global y-limits from both histories
    all_vals = np.concatenate([
        np.array(neural_history[:n_frames]).ravel(),
        np.array(fdm_history[:n_frames]).ravel()
    ])
    ymin, ymax = np.min(all_vals), np.max(all_vals)
    margin = (ymax - ymin) * 0.05 + 0.1
    ymin -= margin
    ymax += margin

    ax_profile, ax_error = axes

    line_fdm, = ax_profile.plot([], [], 'b-', lw=2, label=fdm_label)
    line_neural, = ax_profile.plot([], [], 'r--', lw=2, label=neural_label)
    ax_profile.set_xlim(x[0], x[-1])
    ax_profile.set_ylim(ymin, ymax)
    ax_profile.set_xlabel(xlabel)
    ax_profile.set_ylabel(ylabel)
    ax_profile.legend(loc='upper left')
    ax_profile.grid(True, alpha=0.3)
    step_text = ax_profile.set_title('')

    line_err, = ax_error.semilogy([], [], 'g-', lw=2)
    ax_error.set_xlim(x[0], x[-1])
    ax_error.set_ylim(1e-15, margin * 20)
    ax_error.set_xlabel(xlabel)
    ax_error.set_ylabel(f'|{neural_label} − {fdm_label}|')
    ax_error.set_title('Pointwise Error')
    ax_error.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    def init():
        line_fdm.set_data([], [])
        line_neural.set_data([], [])
        line_err.set_data([], [])
        return line_fdm, line_neural, line_err

    def update(frame):
        nf = neural_history[frame]
        ff = fdm_history[frame]
        line_fdm.set_data(x, ff)
        line_neural.set_data(x, nf)
        err = np.abs(nf - ff) + 1e-15
        line_err.set_data(x, err)
        ax_error.set_ylim(max(err.min() * 0.1, 1e-15), err.max() * 10)
        step_text.set_text(f'Step {frame}/{n_frames - 1}')
        return line_fdm, line_neural, line_err

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=1000 // fps,
                                   blit=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    print(f"  Saved GIF -> {path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Cattaneo-LNO vs Pure FDM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py                         # all benchmarks
  python run_benchmark.py --mode hybrid           # hybrid only
  python run_benchmark.py --mode super-res        # super-resolution only
  python run_benchmark.py --mode frame-gen        # frame generation only
  python run_benchmark.py --mode warm-start       # warm-start only
    python run_benchmark.py --mode rollout          # autoregressive rollout
    python run_benchmark.py --time_over_tau 50      # longer physical horizon
    python run_benchmark.py --num_steps 200         # explicit neural steps
        """,
    )
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'rollout', 'hybrid', 'super-res',
                                 'frame-gen', 'warm-start', 'scaling',
                                 'scaling-evolution'],
                        help='Which benchmark to run (default: all)')
    parser.add_argument('--skip_figures', action='store_true',
                        help='Skip static figure generation during benchmarks')
    parser.add_argument('--skip_gif', action='store_true',
                        help='Skip GIF generation during benchmarks')
    for k, v in DEFAULTS.items():
        if isinstance(v, int):
            parser.add_argument(f'--{k}', type=int, default=v)
        elif isinstance(v, float):
            parser.add_argument(f'--{k}', type=float, default=v)
        elif isinstance(v, str):
            parser.add_argument(f'--{k}', type=str, default=v)

    args = parser.parse_args()

    Path('figures').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    dispatch = {
        'rollout': benchmark_rollout,
        'hybrid': benchmark_hybrid,
        'super-res': benchmark_super_res,
        'frame-gen': benchmark_frame_gen,
        'warm-start': benchmark_warm_start,
        'scaling': benchmark_scaling,
        'scaling-evolution': benchmark_scaling_evolution,
        'all': run_all,
    }
    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
