"""Export rollout reference data for future Athena comparison.

This script runs the trusted Python model and the pure FDM baseline over the
same horizon, saves both histories, and optionally plots a third history from a
future Fortran rollout export.

The Fortran scaffold in this repository currently does not implement a genuine
forward rollout, so the three-way plot only becomes fully available once a
Fortran history file exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_benchmark import (
    DEFAULTS,
    _load,
    _make_T0,
    _resolve_num_steps,
    _run_pure_fdm,
    _warm_up,
    _cuda_sync,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Python/FDM rollout reference data for Athena comparison.")
    parser.add_argument("--grid_size", type=int, default=DEFAULTS["grid_size"])
    parser.add_argument("--dx", type=float, default=DEFAULTS["dx"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    parser.add_argument("--rho_cp", type=float, default=DEFAULTS["rho_cp"])
    parser.add_argument("--bc_left", type=float, default=DEFAULTS["bc_left"])
    parser.add_argument("--bc_right", type=float, default=DEFAULTS["bc_right"])
    parser.add_argument("--timestep_jump", type=int, default=DEFAULTS["timestep_jump"])
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--time_over_tau", type=float, default=50.0)
    parser.add_argument("--spectral_filter", type=str, default=DEFAULTS["spectral_filter"])
    parser.add_argument("--filter_strength", type=float, default=DEFAULTS["filter_strength"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--fortran-history", type=str, default="",
                        help="Optional path to a Fortran rollout file (.npz, .json, or .csv).")
    parser.add_argument("--output-prefix", type=str, default="athena_rollout_comparison",
                        help="Prefix for output files under results/ and figures/.")
    return parser.parse_args()


def run_python_rollout(args: argparse.Namespace) -> dict[str, Any]:
    model, device = _load(args)
    grid_size = args.grid_size
    rollout_steps = _resolve_num_steps(args, model.timestep_jump, default_time_over_tau=args.time_over_tau)
    T0 = _make_T0(grid_size, args.bc_left, args.bc_right)

    _warm_up(model, device, grid_size, args)

    q_t = torch.zeros(1, grid_size, device=device)
    tau_t = torch.full((1, grid_size), args.tau, device=device)
    alpha_t = torch.full((1, grid_size), args.alpha, device=device)
    rho_cp_t = torch.full((1, grid_size), args.rho_cp, device=device)
    bc_l = torch.tensor([args.bc_left], device=device)
    bc_r = torch.tensor([args.bc_right], device=device)
    dt_eff = args.dt
    dt_base = args.dt / max(1, model.timestep_jump)

    T_t = torch.from_numpy(T0).float().unsqueeze(0).to(device)
    T_prev_t = T_t.clone()
    neural_history_gpu = []
    hidden_state = None

    _cuda_sync()
    with torch.inference_mode():
        for _ in range(rollout_steps):
            output = model(
                T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                bc_l, bc_r, dt_eff, args.dx,
                hidden_state=hidden_state,
            )
            T_next = output["T_pred"]
            hidden_state = output.get("hidden_state")
            T_next[:, 0] = args.bc_left
            T_next[:, -1] = args.bc_right
            T_prev_t = T_t
            T_t = T_next
            neural_history_gpu.append(T_t.squeeze(0).detach())

    python_history = np.stack([T0.copy()] + [t.cpu().numpy() for t in neural_history_gpu], axis=0)
    total_fdm_equiv_steps = rollout_steps * model.timestep_jump
    fdm_history, _ = _run_pure_fdm(
        T0, grid_size, args.dx, dt_base, args.tau,
        total_fdm_equiv_steps, args.bc_left, args.bc_right,
        save_every=max(1, model.timestep_jump),
    )

    n_compare = min(len(python_history), len(fdm_history))
    python_history = python_history[:n_compare]
    fdm_history = fdm_history[:n_compare]
    x = np.arange(grid_size)

    python_final = python_history[-1]
    fdm_final = fdm_history[-1]
    rel_err = float(np.linalg.norm(python_final - fdm_final) / (np.linalg.norm(fdm_final) + 1e-15) * 100.0)
    max_err = float(np.max(np.abs(python_final - fdm_final)))

    return {
        "x": x,
        "python_history": python_history,
        "fdm_history": fdm_history,
        "rollout_steps": rollout_steps,
        "timestep_jump": int(model.timestep_jump),
        "physical_time_over_tau": float((rollout_steps * args.dt) / (args.tau + 1e-30)),
        "python_vs_fdm_rel_error_pct": rel_err,
        "python_vs_fdm_max_abs_error": max_err,
    }


def load_fortran_history(path_str: str) -> np.ndarray:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Fortran history file not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
        for key in ("fortran_history", "history", "T_history"):
            if key in data:
                return np.asarray(data[key])
        raise KeyError(f"No rollout history array found in {path}")

    if path.suffix == ".json":
        data = json.loads(path.read_text())
        for key in ("fortran_history", "history", "T_history"):
            if key in data:
                return np.asarray(data[key], dtype=float)
        raise KeyError(f"No rollout history array found in {path}")

    if path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",")

    if path.suffix == ".txt":
        return np.loadtxt(path)

    raise ValueError(f"Unsupported Fortran history format: {path.suffix}")


def save_outputs(args: argparse.Namespace, payload: dict[str, Any], fortran_history: np.ndarray | None) -> tuple[Path, Path]:
    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix
    npz_path = results_dir / f"{prefix}.npz"
    json_path = results_dir / f"{prefix}.json"
    fig_path = figures_dir / f"{prefix}.png"

    save_dict: dict[str, Any] = {
        "x": payload["x"],
        "python_history": payload["python_history"],
        "fdm_history": payload["fdm_history"],
    }
    if fortran_history is not None:
        save_dict["fortran_history"] = fortran_history
    np.savez(npz_path, **save_dict)

    metrics = {
        "rollout_steps": payload["rollout_steps"],
        "timestep_jump": payload["timestep_jump"],
        "physical_time_over_tau": payload["physical_time_over_tau"],
        "python_vs_fdm_rel_error_pct": payload["python_vs_fdm_rel_error_pct"],
        "python_vs_fdm_max_abs_error": payload["python_vs_fdm_max_abs_error"],
        "fortran_history_supplied": fortran_history is not None,
    }
    if fortran_history is not None:
        n_compare = min(len(fortran_history), len(payload["fdm_history"]), len(payload["python_history"]))
        fortran_final = fortran_history[n_compare - 1]
        fdm_final = payload["fdm_history"][n_compare - 1]
        python_final = payload["python_history"][n_compare - 1]
        metrics["fortran_vs_fdm_rel_error_pct"] = float(
            np.linalg.norm(fortran_final - fdm_final) / (np.linalg.norm(fdm_final) + 1e-15) * 100.0
        )
        metrics["fortran_vs_python_rel_error_pct"] = float(
            np.linalg.norm(fortran_final - python_final) / (np.linalg.norm(python_final) + 1e-15) * 100.0
        )
    json_path.write_text(json.dumps(metrics, indent=2))

    plot_histories(payload, fortran_history, fig_path)
    return npz_path, fig_path


def plot_histories(payload: dict[str, Any], fortran_history: np.ndarray | None, fig_path: Path) -> None:
    x = payload["x"]
    python_history = payload["python_history"]
    fdm_history = payload["fdm_history"]

    if fortran_history is not None:
        n_compare = min(len(fortran_history), len(fdm_history), len(python_history))
        fortran_history = fortran_history[:n_compare]
        python_history = python_history[:n_compare]
        fdm_history = fdm_history[:n_compare]

        python_final = python_history[-1]
        fdm_final = fdm_history[-1]
        fortran_final = fortran_history[-1]

        fig, axes = plt.subplots(1, 3, figsize=(19, 5))

        ax = axes[0]
        ax.plot(x, fdm_final, color="navy", lw=2, label="FDM")
        ax.plot(x, python_final, color="crimson", lw=2, ls="--", label="Python NN")
        ax.plot(x, fortran_final, color="darkgreen", lw=2, ls="-.", label="Fortran NN")
        ax.set_title("Final Temperature Profile")
        ax.set_xlabel("Grid point")
        ax.set_ylabel("Temperature [K]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        py_err = [
            np.linalg.norm(py - fdm) / (np.linalg.norm(fdm) + 1e-15) * 100.0
            for py, fdm in zip(python_history[1:], fdm_history[1:])
        ]
        ft_err = [
            np.linalg.norm(ft - fdm) / (np.linalg.norm(fdm) + 1e-15) * 100.0
            for ft, fdm in zip(fortran_history[1:], fdm_history[1:])
        ]
        ax.plot(range(1, len(py_err) + 1), py_err, color="crimson", lw=1.5, label="Python vs FDM")
        ax.plot(range(1, len(ft_err) + 1), ft_err, color="darkgreen", lw=1.5, label="Fortran vs FDM")
        ax.set_title("Rollout Error vs FDM")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Relative error [%]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[2]
        n_show = min(n_compare, 8)
        idx = np.linspace(0, n_compare - 1, n_show, dtype=int)
        for i, step_idx in enumerate(idx):
            alpha = 0.25 + 0.75 * i / max(1, n_show - 1)
            ax.plot(x, fdm_history[step_idx], color="navy", alpha=alpha, lw=0.8)
            ax.plot(x, python_history[step_idx], color="crimson", alpha=alpha, lw=0.8)
            ax.plot(x, fortran_history[step_idx], color="darkgreen", alpha=alpha, lw=0.8)
        ax.set_title("Evolution Snapshots")
        ax.set_xlabel("Grid point")
        ax.set_ylabel("Temperature [K]")
        ax.grid(True, alpha=0.3)

    else:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        python_final = python_history[-1]
        fdm_final = fdm_history[-1]

        ax = axes[0]
        ax.plot(x, fdm_final, color="navy", lw=2, label="FDM")
        ax.plot(x, python_final, color="crimson", lw=2, ls="--", label="Python NN")
        ax.set_title("Final Temperature Profile")
        ax.set_xlabel("Grid point")
        ax.set_ylabel("Temperature [K]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        py_err = [
            np.linalg.norm(py - fdm) / (np.linalg.norm(fdm) + 1e-15) * 100.0
            for py, fdm in zip(python_history[1:], fdm_history[1:])
        ]
        ax.plot(range(1, len(py_err) + 1), py_err, color="crimson", lw=1.5)
        ax.set_title("Python Rollout Error vs FDM")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Relative error [%]")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.axis("off")
        ax.text(
            0.05,
            0.85,
            "Fortran rollout history not supplied.\n\n"
            "The current Fortran executable validates\n"
            "the Athena-compatible module scaffold,\n"
            "but it does not yet implement a genuine\n"
            "numerical forward rollout.\n\n"
            "When a future Fortran history file is\n"
            "available, rerun this script with\n"
            "--fortran-history <file> to get the\n"
            "full three-way plot.",
            va="top",
            ha="left",
            fontsize=11,
        )
        ax.set_title("Fortran Status")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = run_python_rollout(args)
    fortran_history = None
    if args.fortran_history:
        fortran_history = load_fortran_history(args.fortran_history)
    npz_path, fig_path = save_outputs(args, payload, fortran_history)
    print(f"Saved rollout reference: {npz_path}")
    print(f"Saved comparison figure: {fig_path}")
    if fortran_history is None:
        print("No Fortran rollout history supplied; figure includes current scaffold status instead of a third evolution trace.")


if __name__ == "__main__":
    main()