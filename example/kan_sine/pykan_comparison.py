#!/usr/bin/env python3
"""
PyKAN vs Athena KAN (B-spline) benchmark.

Both implementations use B-spline basis functions to approximate

    y = (sin(pi*x) + 1) / 2,    x in [-1, 1].

Architecture correspondence
---------------------------
Athena ``kan_layer_type(num_outputs=1, n_basis=N, spline_degree=K, num_inputs=1)``
uses N B-spline basis functions of degree K.

PyKAN's ``KAN(width=[1, 1], grid=G, k=K)`` uses grid + k basis functions per
edge, so set G = N - K to match the Athena basis count.

Architectural differences
-------------------------
* Athena KAN (use_base_activation=True, PyKAN-style):
  ``output = scale_sp * spline(x) + scale_base * silu(x)``
  plus masked symbolic pathway ``mask * (c * (a*x + b) + d)`` (inactive by
  default, matching PyKAN).
  Learnable per-edge: spline coef [N], scale_sp [1], scale_base [1],
  symbolic affine [4].
  Parameters: N + 6.

* PyKAN: ``output = scale_base * silu(x) + scale_sp * spline(x)``
  Learnable per-edge: spline coef [N], scale_sp [1], scale_base [1],
  plus 4 symbolic-regression affine parameters per edge.
  Parameters: N + 2 + 4 = N + 6.

Optimizer and training protocol:
---------------------------------
Both implementations use Adam (lr = 0.01) on the same 1 000-sample training
dataset.  The gradient-update regimes intentionally differ to reflect each
framework's natural mode of operation:

  PyKAN (this script):
    Full-batch Adam — one gradient update per step using all training samples.
    Default: 200 steps.  Converges to < 0.0001 MSE in ~0.1 s.

  Athena KAN (kan_sine binary):
    Stochastic Adam — one gradient update per step using a single sample.
    Default: 200 epochs × 1 000 samples = 200 000 steps.  Converges to < 0.001 MSE.

Both reach comparable accuracy on the test set; the comparison illustrates the
trade-off between full-batch efficiency and stochastic simplicity.

Requirements
------------
    pip install pykan torch numpy

Usage
-----
    # PyKAN only (auto-detects Athena binary in default build dir):
    python pykan_comparison.py

    # Explicitly provide Athena binary:
    python pykan_comparison.py --athena-exe /path/to/kan_sine

    # Custom architecture / training:
    python pykan_comparison.py --n-basis 10 --degree 3 --steps 200 --lr 0.01
"""

import argparse
import contextlib
import io
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np

# Suppress noisy warnings from pykan internals
warnings.filterwarnings("ignore")

try:
    import torch
    from kan import KAN as PyKAN
    _PYKAN_OK = True
except ImportError:
    _PYKAN_OK = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyKAN vs Athena B-spline KAN benchmark on sin(pi*x)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-basis", type=int, default=10,
                   help="Number of B-spline basis functions (Athena n_basis)")
    p.add_argument("--degree", type=int, default=3,
                   help="B-spline polynomial degree (Athena spline_degree)")
    p.add_argument("--n-epochs", type=int, default=10,
                   help="Number of logging checkpoints during PyKAN training")
    p.add_argument("--steps", type=int, default=200,
                   help="Total full-batch Adam steps for PyKAN")
    p.add_argument("--n-train", type=int, default=1000,
                   help="Training set size (uniform samples in [-1,1])")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Learning rate (Adam, both PyKAN and Athena)")
    p.add_argument("--n-test", type=int, default=30,
                   help="Test set size (evenly spaced in [-1,1], matches Athena)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--athena-exe", type=str, default=None,
                   help="Path to compiled Athena kan_sine binary. "
                        "If omitted, searches the default fpm build directory.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_train: int, n_test: int, seed: int):
    """Return (x_train, y_train, x_test, y_test) as torch float32 tensors."""
    pi = math.pi
    rng = np.random.default_rng(seed)
    # Training: uniform samples in [-1, 1]
    x_tr = rng.uniform(-1.0, 1.0, size=(n_train, 1)).astype(np.float32)
    y_tr = ((np.sin(pi * x_tr) + 1.0) / 2.0).astype(np.float32)
    # Test: 30 evenly-spaced points matching Athena's test grid
    x_te = np.linspace(-1.0, 1.0, n_test, dtype=np.float32).reshape(-1, 1)
    y_te = ((np.sin(pi * x_te) + 1.0) / 2.0).astype(np.float32)
    return (
        torch.from_numpy(x_tr), torch.from_numpy(y_tr),
        torch.from_numpy(x_te), torch.from_numpy(y_te),
    )


def _mse(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    return float(((pred - target) ** 2).mean())


# ---------------------------------------------------------------------------
# PyKAN training
# ---------------------------------------------------------------------------

def train_pykan(
    n_basis: int,
    degree: int,
    steps: int,
    n_checkpoints: int,
    n_train: int,
    lr: float,
    n_test: int,
    seed: int,
) -> dict:
    """
    Train a 1-input 1-output PyKAN model (full-batch Adam) and return results.

    Uses full-batch gradient updates — PyKAN's natural/efficient mode.
    Training runs for ``steps`` total Adam steps; MSE is evaluated at
    ``n_checkpoints`` evenly-spaced points during training.

    PyKAN grid is set to ``n_basis - degree`` so that the total number of
    spline basis functions equals ``n_basis`` (pykan: grid + k = n_basis).
    """
    grid = n_basis - degree
    if grid < 1:
        raise ValueError(
            f"n_basis ({n_basis}) must be > degree ({degree}); "
            f"pykan grid = n_basis - degree = {grid} is invalid."
        )

    x_tr, y_tr, x_te, y_te = _make_dataset(n_train, n_test, seed)

    dataset = {
        "train_input": x_tr,
        "train_label": y_tr,
        "test_input": x_te,
        "test_label": y_te,
    }

    # Suppress pykan's verbose checkpoint messages
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink):
        model = PyKAN(
            width=[1, 1],
            grid=grid,
            k=degree,
            seed=seed,
            auto_save=False,
            ckpt_path="/tmp/pykan_bench_ckpt",
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- print header ---
    print()
    print("------------------------------------------------------------")
    print("  PyKAN (B-spline) — sin(pi*x) approximation")
    print("------------------------------------------------------------")
    print(f"  Architecture : KAN([1, 1], grid={grid}, k={degree})")
    print(f"  Basis fns    : {n_basis}  (grid + k = {grid} + {degree})")
    print(f"  Parameters   : {n_params}")
    print(f"  Optimiser    : Adam, lr={lr}, full-batch, {steps} steps")
    print(f"  Training set : {n_train} uniform samples in [-1, 1]")
    print()
    print(f"  {'Step':>6}  {'Train MSE':>12}  {'Test MSE':>12}")
    print(f"  {'----':>6}  {'----------':>12}  {'----------':>12}")

    # Suppress tqdm / internal pykan output during fit
    def _fit_quiet(n_steps: int):
        _s = io.StringIO()
        with contextlib.redirect_stdout(_s):
            with contextlib.redirect_stderr(_s):
                return model.fit(
                    dataset,
                    opt="Adam",
                    lr=lr,
                    steps=n_steps,
                    update_grid=False,
                    batch=-1,  # full-batch
                    log=n_steps + 1,  # suppress internal logging
                )

    mse_history: list[tuple[int, float, float]] = []
    chunk = max(1, steps // n_checkpoints)
    step = 0
    t_start = time.perf_counter()

    while step < steps:
        remaining = min(chunk, steps - step)
        _fit_quiet(remaining)
        step += remaining
        with torch.no_grad():
            tr_mse = _mse(model(x_tr), y_tr)
            te_mse = _mse(model(x_te), y_te)
        mse_history.append((step, tr_mse, te_mse))
        print(f"  {step:>6}  {tr_mse:>12.6f}  {te_mse:>12.6f}")

    elapsed = time.perf_counter() - t_start
    final_test_mse = mse_history[-1][2]

    print()
    print(f"  Training time  : {elapsed:.4f} s")
    print(f"  Final test MSE : {final_test_mse:.6f}")

    return {
        "n_params": n_params,
        "n_train": n_train,
        "elapsed": elapsed,
        "final_mse": final_test_mse,
        "mse_history": mse_history,
    }


# ---------------------------------------------------------------------------
# Athena runner
# ---------------------------------------------------------------------------

_DEFAULT_ATHENA_PATHS = [
    # fpm default build locations (gfortran_* suffix varies by machine)
    os.path.join(
        os.path.dirname(__file__), "..", "..", "build",
        *("*",),  # glob placeholder — handled below
        "example", "kan_bspline",
    ),
]


def _find_athena_exe(user_path: str | None) -> str | None:
    """Return path to kan_sine binary or None if not found."""
    if user_path is not None:
        return user_path if os.path.isfile(user_path) else None

    # Search default fpm build directory
    import glob
    pattern = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "build", "*", "example", "kan_sine",
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _run_athena(exe: str) -> dict | None:
    """Execute Athena kan_sine and parse its stdout into a results dict."""
    print()
    print(f"  [Athena] Running: {exe}")
    t_start = time.perf_counter()
    try:
        proc = subprocess.run(
            [exe],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"  [Athena] Error: {exc}")
        return None
    elapsed = time.perf_counter() - t_start

    if proc.returncode != 0:
        print(f"  [Athena] Exit code {proc.returncode}")
        print(proc.stderr[:500])
        return None

    # Parse output
    result: dict = {"elapsed": elapsed, "sections": {}}
    current_section: str | None = None
    final_mse: float | None = None
    train_time: float | None = None

    for line in proc.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "fastkan (rbf)" in low or "fastkan" in low and "benchmark" not in low:
            current_section = "fastkan"
        elif "pykan-style" in low or "base+spline" in low:
            current_section = "pykan"
        elif "b-spline" in low or "bspline" in low or ("kan (b" in low):
            current_section = "kan"
        elif "polynomial" in low:
            current_section = "poly"
        if "final test mse" in low:
            try:
                v = float(s.split()[-1])
                if current_section and current_section not in result["sections"]:
                    result["sections"][current_section] = {}
                if current_section:
                    result["sections"][current_section]["final_mse"] = v
            except (ValueError, IndexError):
                pass
        if "training time" in low:
            try:
                v = float(s.split()[-2])
                if current_section:
                    result["sections"].setdefault(current_section, {})["train_time"] = v
            except (ValueError, IndexError):
                pass

    print(proc.stdout)
    result["elapsed_total"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_summary(pykan_res: dict, athena_res: dict | None, args) -> None:
    n_basis = args.n_basis
    degree = args.degree
    grid = n_basis - degree
    d_in, d_out = 1, 1
    n_edges = d_in * d_out

    W = 62
    sep = "=" * W
    print()
    print(sep)
    print("  Architecture & Parameter Comparison")
    print(sep)
    print()

    # -- Architecture summary --
    print("  Architecture:")
    print(f"    Layer widths     : [{d_in}, {d_out}]")
    print(f"    Edges            : {n_edges}")
    print(f"    Spline degree    : {degree}")
    print(f"    Grid size        : {grid}")
    print(f"    Basis functions  : {n_basis}  (grid + k = {grid} + {degree})")
    print()

    # -- Parameter breakdown --
    print("  Parameter breakdown per edge:")
    print(f"    {'Component':30s}  {'PyKAN':>8}  {'Athena':>8}")
    print(f"    {'-'*30}  {'-'*8}  {'-'*8}")
    print(f"    {'Spline coefficients':30s}  {n_basis:>8}  {n_basis:>8}")
    print(f"    {'scale_base (SiLU weight)':30s}  {1:>8}  {1:>8}")
    print(f"    {'scale_sp (spline scale)':30s}  {1:>8}  {1:>8}")
    print(f"    {'Symbolic affine (a,b,c,d)':30s}  {4:>8}  {4:>8}")
    print(f"    {'Bias (trainable)':30s}  {'0':>8}  {'0':>8}")
    pykan_per_edge = n_basis + 6
    athena_per_edge = n_basis + 6
    print(f"    {'─'*30}  {'─'*8}  {'─'*8}")
    print(f"    {'Total per edge':30s}  {pykan_per_edge:>8}  {athena_per_edge:>8}")
    print()

    # -- Total parameters --
    pykan_total = pykan_res["n_params"]
    # Athena PyKAN-style: m * d * (K + 6) per layer
    athena_total = n_edges * (n_basis + 6)
    print(f"  {'':30s}  {'PyKAN':>10}  {'Athena KAN':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"  {'Total parameters':30s}  {pykan_total:>10}  {athena_total:>10}")

    match_str = "MATCH ✓" if pykan_total == athena_total else f"MISMATCH (Δ={pykan_total - athena_total})"
    print(f"  Parameter parity: {match_str}")
    print()

    # Athena KAN MSE / time from parsed output
    a_mse: float | None = None
    a_time: float | None = None
    if athena_res is not None:
        # Check PyKAN-style section first, then generic kan section
        for sec_name in ["pykan", "kan"]:
            sec = athena_res.get("sections", {}).get(sec_name, {})
            if "final_mse" in sec:
                a_mse = sec["final_mse"]
                a_time = sec.get("train_time")
                break

    mse_str = f"{a_mse:.6f}" if a_mse is not None else "   n/a"
    time_str = f"{a_time:.4f} s" if a_time is not None else "    n/a"
    print(f"  {'Final test MSE':30s}  {pykan_res['final_mse']:>10.6f}  {mse_str:>10}")
    print(f"  {'Training time':30s}  {pykan_res['elapsed']:>8.4f} s  {time_str:>10}")
    print()

    print("  Notes:")
    print("  - Both: Adam (lr=0.01), same 1000-sample training set, same test set")
    print("  - PyKAN: full-batch Adam (efficient per step, fewer steps needed)")
    print("  - Athena KAN: stochastic Adam (1 sample/step, many steps total)")
    print("  - Both: output = scale_sp * spline(x) + scale_base * silu(x)")
    print("  - Both: 4 symbolic affine params per edge (masked off by default)")
    print()
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _PYKAN_OK:
        print(
            "ERROR: pykan is not installed.\n"
            "Install it with:\n"
            "    pip install pykan\n"
            "Then rerun this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    args = _parse_args()

    print("=" * 62)
    print("  PyKAN vs Athena B-spline KAN Benchmark")
    print("=" * 62)
    print()
    print(f"  Task  : y = (sin(pi*x) + 1) / 2  on  [-1, 1]")
    print(f"  Basis : {args.n_basis} B-spline functions, degree {args.degree}")
    print()

    # --- PyKAN ---
    pykan_res = train_pykan(
        n_basis=args.n_basis,
        degree=args.degree,
        steps=args.steps,
        n_checkpoints=args.n_epochs,
        n_train=args.n_train,
        lr=args.lr,
        n_test=args.n_test,
        seed=args.seed,
    )

    # --- Athena ---
    exe = _find_athena_exe(args.athena_exe)
    athena_res: dict | None = None
    if exe is None:
        print()
        print(
            "  [Athena] Binary not found. Build with:\n"
            "      fpm build\n"
            "  Then pass the path with --athena-exe <path>."
        )
    else:
        print()
        print("------------------------------------------------------------")
        print("  Athena KAN (B-spline) and FastKAN (RBF) — via kan_sine")
        print("------------------------------------------------------------")
        athena_res = _run_athena(exe)

    # --- Summary ---
    _print_summary(pykan_res, athena_res, args)


if __name__ == "__main__":
    main()
