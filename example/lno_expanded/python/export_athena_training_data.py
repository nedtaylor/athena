"""Export supervised training data for Athena/Fortran.

The Fortran training path consumes the same exact-BC projection and pair
clipping used by the Python trainer, along with per-sample BC/tau metadata and
the full trajectory sets needed for rollout supervision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from train_cattaneo_lno import _clip_extreme_samples, _project_dataset_to_exact_bc


def to_numpy(tensor_or_value):
    if isinstance(tensor_or_value, torch.Tensor):
        return tensor_or_value.detach().cpu().numpy()
    return np.asarray(tensor_or_value)


def to_sample_scalar(array_like):
    array = to_numpy(array_like)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array[:, 0]
    raise ValueError(f"Expected rank-1 or rank-2 sample metadata, got shape {array.shape}")


def save_vector(path: Path, values) -> None:
    np.savetxt(path, np.asarray(values, dtype=np.float32).reshape(-1, 1), fmt="%.8e")


def save_trajectory_set(path: Path, trajectories: np.ndarray) -> None:
    flat = trajectories.reshape(-1, trajectories.shape[-1]).astype(np.float32)
    np.savetxt(path, flat, fmt="%.8e")


def main() -> None:
    root = Path(__file__).resolve().parent
    data = torch.load(root / "data" / "training_data.pt", map_location="cpu", weights_only=False)

    out_dir = root.parent / "Fortran" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _clip_extreme_samples(_project_dataset_to_exact_bc(data["train"]))
    val = _clip_extreme_samples(_project_dataset_to_exact_bc(data["val"]))
    train_traj_data = _project_dataset_to_exact_bc(data["train_trajectories"])
    val_traj_data = _project_dataset_to_exact_bc(data["val_trajectories"])

    train_inputs = np.concatenate([
        to_numpy(train["T_n"]),
        to_numpy(train["T_nm1"]),
    ], axis=1).astype(np.float32)
    train_targets = to_numpy(train["T_target"]).astype(np.float32)

    val_inputs = np.concatenate([
        to_numpy(val["T_n"]),
        to_numpy(val["T_nm1"]),
    ], axis=1).astype(np.float32)
    val_targets = to_numpy(val["T_target"]).astype(np.float32)

    np.savetxt(out_dir / "train_inputs.txt", train_inputs, fmt="%.8e")
    np.savetxt(out_dir / "train_targets.txt", train_targets, fmt="%.8e")
    np.savetxt(out_dir / "val_inputs.txt", val_inputs, fmt="%.8e")
    np.savetxt(out_dir / "val_targets.txt", val_targets, fmt="%.8e")

    save_vector(out_dir / "train_tau.txt", to_sample_scalar(train["tau"]))
    save_vector(out_dir / "train_alpha.txt", to_sample_scalar(train["alpha"]))
    save_vector(out_dir / "train_bc_left.txt", to_sample_scalar(train["bc_left"]))
    save_vector(out_dir / "train_bc_right.txt", to_sample_scalar(train["bc_right"]))
    save_vector(out_dir / "val_tau.txt", to_sample_scalar(val["tau"]))
    save_vector(out_dir / "val_alpha.txt", to_sample_scalar(val["alpha"]))
    save_vector(out_dir / "val_bc_left.txt", to_sample_scalar(val["bc_left"]))
    save_vector(out_dir / "val_bc_right.txt", to_sample_scalar(val["bc_right"]))

    train_traj = to_numpy(train_traj_data["trajectories"]).astype(np.float32)
    val_traj = to_numpy(val_traj_data["trajectories"]).astype(np.float32)
    save_trajectory_set(out_dir / "train_trajectories.txt", train_traj)
    save_trajectory_set(out_dir / "val_trajectories.txt", val_traj)
    np.savetxt(out_dir / "val_trajectory_0.txt", val_traj[0], fmt="%.8e")
    np.savetxt(out_dir / "train_trajectory_0.txt", train_traj[0], fmt="%.8e")

    save_vector(out_dir / "train_trajectory_tau.txt", to_sample_scalar(train_traj_data["tau"]))
    save_vector(out_dir / "train_trajectory_alpha.txt", to_sample_scalar(train_traj_data["alpha"]))
    save_vector(out_dir / "train_trajectory_bc_left.txt", to_sample_scalar(train_traj_data["bc_left"]))
    save_vector(out_dir / "train_trajectory_bc_right.txt", to_sample_scalar(train_traj_data["bc_right"]))
    save_vector(out_dir / "val_trajectory_tau.txt", to_sample_scalar(val_traj_data["tau"]))
    save_vector(out_dir / "val_trajectory_alpha.txt", to_sample_scalar(val_traj_data["alpha"]))
    save_vector(out_dir / "val_trajectory_bc_left.txt", to_sample_scalar(val_traj_data["bc_left"]))
    save_vector(out_dir / "val_trajectory_bc_right.txt", to_sample_scalar(val_traj_data["bc_right"]))

    metadata = {
        "grid_size": int(train["T_n"].shape[1]),
        "input_dim": int(train_inputs.shape[1]),
        "output_dim": int(train_targets.shape[1]),
        "n_train": int(train_inputs.shape[0]),
        "n_val": int(val_inputs.shape[0]),
        "n_train_trajectories": int(train_traj.shape[0]),
        "n_val_trajectories": int(val_traj.shape[0]),
        "trajectory_length": int(val_traj.shape[1]),
        "dt": float(train["dt"]),
        "dx": float(train["dx"]),
        "bc_left": float(to_sample_scalar(val_traj_data["bc_left"])[0]),
        "bc_right": float(to_sample_scalar(val_traj_data["bc_right"])[0]),
        "tau": float(to_sample_scalar(val_traj_data["tau"])[0]),
        "alpha": float(to_sample_scalar(val_traj_data["alpha"])[0]),
        "rho_cp": float(to_numpy(val_traj_data["rho_cp"])[0, 0]),
        "rollout_steps": int(val_traj.shape[1] - 1),
        "temp_ref": 200.0,
        "delta_t": 100.0,
        "reference_trajectory_file": "val_trajectory_0.txt",
        "reference_train_trajectory_file": "train_trajectory_0.txt",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Exported Athena training data to {out_dir}")


if __name__ == "__main__":
    main()