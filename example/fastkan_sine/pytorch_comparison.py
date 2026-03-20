#!/usr/bin/env python3
"""
PyTorch equivalent of the athena kan_sine example, with benchmarking.

Implements a single KAN (Kolmogorov-Arnold Network) layer with RBF activations
to approximate y = (sin(x) + 1) / 2 over [0, 2*pi].

Architecture matches athena kan_layer_type:
    phi_{i,k}(x_i) = exp(-0.5 * ((x_i - c_{i,k}) / sigma_{i,k})^2)
    y_j = sum_{i,k} W_{j,i,k} * phi_{i,k}(x_i) + b_j

Initialization exactly matches athena init_kan:
    centres   : uniformly spaced over [-1, 1] for each input dimension
    bandwidths: (centre_max - centre_min) / (n_basis - 1)
    weights   : Glorot uniform  U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    bias      : zero

Optimizer: vanilla SGD (momentum=0, lr=0.01), matching athena sgd_optimiser_type.
Loss:       MSE, matching athena "mse" loss method.

Usage
-----
# PyTorch only (will not attempt to compare with athena):
    python kan_sine_pytorch.py

# Full benchmark against a compiled athena binary:
    python kan_sine_pytorch.py --athena-exe /path/to/kan_sine
"""

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# KAN Layer
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    """
    KAN layer with trainable RBF activations.

    Trainable parameters (matches athena params(1..4)):
        centres     shape [num_inputs * n_basis]
        bandwidths  shape [num_inputs * n_basis]
        weights     shape [num_outputs, num_inputs * n_basis]
        bias        shape [num_outputs]  (optional)
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        n_basis: int = 8,
        use_bias: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.n_basis = n_basis
        self.use_bias = use_bias

        dK = num_inputs * n_basis

        self.centres = nn.Parameter(torch.empty(dK))
        self.bandwidths = nn.Parameter(torch.empty(dK))
        self.weights = nn.Parameter(torch.empty(num_outputs, dK))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_outputs))
        else:
            self.register_parameter("bias", None)

        self._init_params(seed)

    def _init_params(self, seed: int) -> None:
        """
        Initialise parameters to match athena init_kan.

        Centres and bandwidths are fully deterministic. Weights use Glorot
        uniform with numpy seed *seed* -- this replicates athena's
        ``random_seed(put=[seed, seed, ...])`` convention at a statistical
        level (same distribution, same seed value) though the underlying RNG
        stream will differ between Fortran and Python.
        """
        K = self.n_basis
        dK = self.num_inputs * K

        # ------------------------------------------------------------------
        # Centres: uniformly spaced over [-1, 1] per input dimension.
        # Athena: params(1)%val((i-1)*K + k, 1) = -1 + 2*(k-1)/(K-1)
        # ------------------------------------------------------------------
        c_vals = np.linspace(-1.0, 1.0, K, dtype=np.float32)
        centres = np.tile(c_vals, self.num_inputs)   # shape [dK]
        self.centres.data.copy_(torch.from_numpy(centres))

        # ------------------------------------------------------------------
        # Bandwidths: spacing between adjacent centres.
        # Athena: params(2)%val(:,1) = (centre_max - centre_min) / (K - 1)
        # ------------------------------------------------------------------
        sigma = (2.0 / (K - 1)) if K > 1 else 1.0
        self.bandwidths.data.fill_(float(sigma))

        # ------------------------------------------------------------------
        # Weights: Glorot uniform.
        # Athena: glorot_uniform_initialise with fan_in=dK, fan_out=m.
        #   limit = sqrt(6 / (fan_in + fan_out))
        #   W ~ U(-limit, limit)
        # ------------------------------------------------------------------
        fan_in = dK
        fan_out = self.num_outputs
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        rng = np.random.default_rng(seed)
        w = rng.uniform(-limit, limit, size=(self.num_outputs, dK)).astype(np.float32)
        self.weights.data.copy_(torch.from_numpy(w))

        # bias is already zero from torch.zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching athena forward_kan.

        Args:
            x: shape [batch, num_inputs]
        Returns:
            shape [batch, num_outputs]
        """
        # Expand input: replicate each x_i n_basis times -> [batch, dK].
        # Matches expand_matrix @ x in athena (expand_matrix repeats each
        # input component across all K basis rows for that dimension).
        x_expanded = x.repeat_interleave(self.n_basis, dim=1)   # [batch, dK]

        # RBF activations:  phi = exp(-0.5 * ((x - c) / sigma)^2)
        diff = x_expanded - self.centres                          # [batch, dK]
        phi = torch.exp(-0.5 * (diff / self.bandwidths) ** 2)    # [batch, dK]

        # Linear combination:  out = phi @ W^T + bias
        out = phi @ self.weights.T                                 # [batch, m]
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Deterministic test data
# ---------------------------------------------------------------------------

def make_test_data(test_size: int = 30) -> tuple:
    """
    Deterministic test set, matching athena kan_sine exactly.

    Athena:
        x_test(1,i) = (i-1) * 2*pi / test_size   for i=1..test_size
        y_test(1,i) = (sin(x_test(1,i)) + 1) / 2
    """
    pi = math.pi
    x = np.array(
        [i * 2.0 * pi / test_size for i in range(test_size)],
        dtype=np.float32,
    )
    y = (np.sin(x) + 1.0) / 2.0
    return x, y


# ---------------------------------------------------------------------------
# PyTorch training
# ---------------------------------------------------------------------------

def train_pytorch(
    num_iterations: int = 10_000,
    n_basis: int = 10,
    lr: float = 0.01,
    seed: int = 42,
    print_interval: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Train KAN to approximate (sin(x)+1)/2 using vanilla SGD.

    Training samples are drawn uniformly from [0, 2*pi] in the same order as
    athena (both use seed 42; Fortran and Python draw from the same half-open
    uniform distribution [0,1) though their internal streams differ).

    Returns a results dict for benchmarking.
    """
    # Seed both torch (for any dropout/stochastic ops) and numpy (training data)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = KANLayer(
        num_inputs=1, num_outputs=1, n_basis=n_basis, use_bias=True, seed=seed
    )
    # Vanilla SGD with no momentum -- matches athena sgd_optimiser_type defaults
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.MSELoss()

    x_test_np, y_test_np = make_test_data()
    x_test = torch.from_numpy(x_test_np).unsqueeze(1)   # [30, 1]
    y_test = torch.from_numpy(y_test_np).unsqueeze(1)   # [30, 1]

    if verbose:
        print("Sine function approximation using a KAN layer (PyTorch)")
        print("----------------------------------------------------------")
        print()
        print("Training network")
        print("----------------")
        print(f"{'Iteration':>10}{'Test MSE':>12}")

    mse_history: dict[int, float] = {}

    t_start = time.perf_counter()

    for n in range(num_iterations + 1):
        # Training sample: x ~ U(0, 2*pi),  y = (sin(x)+1)/2
        # Athena draws one real via random_number(x), scales to [0, 2*pi].
        pi = math.pi
        x_val = float(rng.random(dtype=np.float32)) * 2.0 * pi
        y_val = (math.sin(x_val) + 1.0) / 2.0

        x_t = torch.tensor([[x_val]], dtype=torch.float32)
        y_t = torch.tensor([[y_val]], dtype=torch.float32)

        optim.zero_grad()
        pred = model(x_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optim.step()

        if n % print_interval == 0:
            with torch.no_grad():
                y_pred = model(x_test)
                test_mse = float(criterion(y_pred, y_test))
            mse_history[n] = test_mse
            if verbose:
                print(f"{n:>10}{test_mse:>12.6f}")

    elapsed = time.perf_counter() - t_start

    # Final predictions on test set
    with torch.no_grad():
        y_pred_np = model(x_test).squeeze(1).numpy()

    final_mse = float(np.mean((y_pred_np - y_test_np) ** 2))

    if verbose:
        print()
        print("Final predictions vs targets:")
        print(f"{'x':>10}{'target':>12}{'predicted':>12}")
        for i in range(len(x_test_np)):
            print(
                f"{x_test_np[i]:>10.4f}"
                f"{y_test_np[i]:>12.4f}"
                f"{y_pred_np[i]:>12.4f}"
            )
        print()
        print(f" Final test MSE:  {final_mse:.6f}")

    return {
        "elapsed": elapsed,
        "mse_history": mse_history,
        "final_mse": final_mse,
        "x_test": x_test_np,
        "y_test": y_test_np,
        "y_pred": y_pred_np,
    }


# ---------------------------------------------------------------------------
# Athena runner
# ---------------------------------------------------------------------------

def _parse_athena_output(text: str) -> dict:
    """Parse stdout from athena kan_sine into the same results dict format."""
    mse_history: dict[int, float] = {}
    final_mse: float | None = None
    x_list: list[float] = []
    y_target_list: list[float] = []
    y_pred_list: list[float] = []

    in_predictions = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Final MSE line:  " Final test MSE:  0.001234"
        if "Final test MSE" in line:
            try:
                final_mse = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
            in_predictions = False
            continue

        # Header of predictions table
        if "x" in line and "target" in line and "predicted" in line:
            in_predictions = True
            continue

        if in_predictions:
            parts = line.split()
            if len(parts) == 3:
                try:
                    x_list.append(float(parts[0]))
                    y_target_list.append(float(parts[1]))
                    y_pred_list.append(float(parts[2]))
                    continue
                except ValueError:
                    in_predictions = False

        # Iteration/MSE table rows: two columns (int, float)
        if not in_predictions:
            parts = line.split()
            if len(parts) == 2:
                try:
                    it = int(parts[0])
                    mse = float(parts[1])
                    mse_history[it] = mse
                except ValueError:
                    pass

    return {
        "final_mse": final_mse,
        "mse_history": mse_history,
        "x_test": np.array(x_list, dtype=np.float32),
        "y_test": np.array(y_target_list, dtype=np.float32),
        "y_pred": np.array(y_pred_list, dtype=np.float32),
    }


def run_athena(executable: str, verbose: bool = True) -> dict | None:
    """
    Execute the compiled athena kan_sine binary, parse its output.
    Returns None if the executable is not available.
    """
    if not os.path.isfile(executable):
        print(f"[athena] Executable not found: {executable}")
        return None

    if verbose:
        print(f"\n[athena] Running: {executable}")

    t_start = time.perf_counter()
    try:
        proc = subprocess.run(
            [executable],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("[athena] Process timed out (>600 s).")
        return None

    elapsed = time.perf_counter() - t_start

    if proc.returncode != 0:
        print(f"[athena] Non-zero exit code {proc.returncode}:")
        print(proc.stderr)
        return None

    if verbose:
        print(proc.stdout)

    result = _parse_athena_output(proc.stdout)
    result["elapsed"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Benchmark summary
# ---------------------------------------------------------------------------

def print_benchmark(pytorch_res: dict, athena_res: dict | None) -> None:
    """Print a side-by-side benchmark summary."""
    sep = "=" * 66
    print()
    print(sep)
    print("BENCHMARK SUMMARY")
    print(sep)

    has_athena = athena_res is not None

    header = f"{'Metric':<34}{'PyTorch':>16}"
    if has_athena:
        header += f"{'Athena':>16}"
    print(header)
    print("-" * 66)

    def row(label: str, py_val: float, at_val: float | None = None, fmt: str = ".3f") -> str:
        s = f"{label:<34}{py_val:>16{fmt}}"
        if has_athena:
            s += f"{at_val:>16{fmt}}" if at_val is not None else f"{'N/A':>16}"
        return s

    print(row("Training time (s)", pytorch_res["elapsed"],
               athena_res["elapsed"] if has_athena else None))
    print(row("Final test MSE", pytorch_res["final_mse"],
               athena_res["final_mse"] if has_athena else None, fmt=".6f"))

    if has_athena and athena_res["elapsed"] > 0:
        speedup = athena_res["elapsed"] / pytorch_res["elapsed"]
        print(f"\n{'Speedup (athena/pytorch)':<34}{speedup:>16.2f}x")

    # Per-checkpoint convergence
    if pytorch_res.get("mse_history"):
        print()
        if has_athena and athena_res.get("mse_history"):
            print(f"  {'Iteration':>10}  {'PyTorch MSE':>14}  {'Athena MSE':>14}  {'Ratio':>8}")
            for it in sorted(pytorch_res["mse_history"]):
                py = pytorch_res["mse_history"][it]
                at = athena_res["mse_history"].get(it)
                ratio_str = f"{at / py:>8.3f}" if (at and py) else f"{'N/A':>8}"
                at_str = f"{at:>14.6f}" if at is not None else f"{'N/A':>14}"
                print(f"  {it:>10}  {py:>14.6f}  {at_str}  {ratio_str}")
        else:
            print(f"  {'Iteration':>10}  {'PyTorch MSE':>14}")
            for it, mse in sorted(pytorch_res["mse_history"].items()):
                print(f"  {it:>10}  {mse:>14.6f}")

    # Prediction comparison
    if has_athena and len(athena_res["y_pred"]) > 0 and len(pytorch_res["y_pred"]) > 0:
        n = min(len(pytorch_res["y_pred"]), len(athena_res["y_pred"]))
        max_diff = float(np.max(np.abs(pytorch_res["y_pred"][:n] - athena_res["y_pred"][:n])))
        mean_diff = float(np.mean(np.abs(pytorch_res["y_pred"][:n] - athena_res["y_pred"][:n])))
        print()
        print(f"  Max  |PyTorch_pred - Athena_pred| : {max_diff:.6f}")
        print(f"  Mean |PyTorch_pred - Athena_pred| : {mean_diff:.6f}")

    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KAN sine approximation benchmark: PyTorch vs athena (Fortran)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--athena-exe",
        default=None,
        metavar="PATH",
        help="Path to the compiled athena kan_sine binary.",
    )
    parser.add_argument("--num-iterations", type=int, default=10_000)
    parser.add_argument("--n-basis", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration output; show only the benchmark summary.",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    sep = "=" * 66
    print(sep)
    print("KAN Sine Approximation: PyTorch vs Athena (Fortran) Benchmark")
    print(sep)
    print(f"  Architecture : num_inputs=1, num_outputs=1, n_basis={args.n_basis}")
    print(f"  Optimiser    : SGD (vanilla), lr={args.lr}")
    print(f"  Iterations   : {args.num_iterations}")
    print(f"  Seed         : {args.seed}")
    print(f"  Target       : y = (sin(x) + 1) / 2  over [0, 2*pi]")
    print(sep)
    print()

    # ------------------------------------------------------------------
    # PyTorch run
    # ------------------------------------------------------------------
    print("[pytorch] Starting PyTorch training ...")
    pytorch_res = train_pytorch(
        num_iterations=args.num_iterations,
        n_basis=args.n_basis,
        lr=args.lr,
        seed=args.seed,
        print_interval=1000,
        verbose=verbose,
    )
    print(f"\n[pytorch] Done. elapsed={pytorch_res['elapsed']:.3f}s  "
          f"final_MSE={pytorch_res['final_mse']:.6f}")

    # ------------------------------------------------------------------
    # Athena run (optional)
    # ------------------------------------------------------------------
    athena_res = None
    if args.athena_exe:
        athena_res = run_athena(args.athena_exe, verbose=verbose)
        if athena_res:
            print(f"[athena]  Done. elapsed={athena_res['elapsed']:.3f}s  "
                  f"final_MSE={athena_res['final_mse']:.6f}")
    else:
        print("\n[info] No --athena-exe provided; skipping Fortran comparison.")
        print("[info] Build athena with `fpm build` and pass the binary path "
              "via --athena-exe to enable the benchmark.")

    # ------------------------------------------------------------------
    # Benchmark summary
    # ------------------------------------------------------------------
    print_benchmark(pytorch_res, athena_res)


if __name__ == "__main__":
    main()
