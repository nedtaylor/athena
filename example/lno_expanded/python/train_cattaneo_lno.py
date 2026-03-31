"""
Training Script for Cattaneo-LNO
================================

Trains the Cattaneo-LNO model with physics-informed loss.
Optimized hyperparameters for hyperbolic PDEs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import random
import time
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse

from cattaneo_lno import (create_cattaneo_model, CattaneoPhysicsLoss,
                          AdaptiveLossWeights, DimensionlessScaler)
from data_generation import generate_training_data


# ---------------------------------------------------------------------------
# GPU-native batch iterators (no DataLoader overhead)
# ---------------------------------------------------------------------------

class GPUBatchLoader:
    """Pre-stacks all pair data as contiguous GPU tensors.

    Each epoch, call `shuffle()` to randomise order, then iterate to
    get dict batches.  No per-sample indexing, no collation.
    """
    TENSOR_KEYS = ['T_n', 'T_nm1', 'T_target', 'T_prev', 'T_prev2',
                   'q', 'tau', 'alpha', 'rho_cp', 'bc_left', 'bc_right']

    def __init__(self, data: Dict, batch_size: int, device: str = 'cpu', shuffle: bool = True):
        required = ['T_n', 'T_nm1', 'T_target', 'T_prev', 'T_prev2',
                    'q', 'tau', 'alpha', 'rho_cp', 'bc_left', 'bc_right', 'dt', 'dx']
        for k in required:
            if k not in data:
                raise ValueError(f"Missing required key: {k}")

        self.device = device
        self.batch_size = batch_size
        self.do_shuffle = shuffle

        # Pre-stack on device as contiguous tensors
        self.tensors = {}
        for k in self.TENSOR_KEYS:
            v = data[k]
            self.tensors[k] = v.to(device).contiguous() if isinstance(v, torch.Tensor) else v
        # Alias T_target -> T for loss compatibility
        self.tensors['T'] = self.tensors['T_target']
        self.dt = data['dt']
        self.dx = data['dx']
        self.n = len(self.tensors['T_n'])
        self._perm = torch.arange(self.n, device=device)

    def shuffle(self):
        if self.do_shuffle:
            self._perm = torch.randperm(self.n, device=self.device)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        for start in range(0, self.n, self.batch_size):
            idx = self._perm[start:start + self.batch_size]
            batch = {k: v[idx] for k, v in self.tensors.items()}
            batch['dt'] = self.dt
            batch['dx'] = self.dx
            yield batch


class GPUTrajectoryLoader:
    """Pre-stacks trajectory data on GPU and yields random windows."""

    def __init__(self, traj_data: Dict, window_size: int, batch_size: int,
                 device: str = 'cpu', shuffle: bool = True):
        _to = lambda t: t.to(device).contiguous() if isinstance(t, torch.Tensor) else t
        self.trajectories = _to(traj_data['trajectories'])  # [N, L, G]
        self.tau = _to(traj_data['tau'])
        self.alpha = _to(traj_data['alpha'])
        self.rho_cp = _to(traj_data['rho_cp'])
        self.q = _to(traj_data['q'])
        self.bc_left = _to(traj_data['bc_left'])
        self.bc_right = _to(traj_data['bc_right'])
        self.dt = traj_data['dt']
        self.dx = traj_data['dx']
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        self.do_shuffle = shuffle

        n_traj, seq_len, _ = self.trajectories.shape
        self.wpt = max(1, seq_len - window_size + 1)  # windows per traj
        self.n = n_traj * self.wpt
        self._perm = torch.arange(self.n, device=device)

    def shuffle(self):
        if self.do_shuffle:
            self._perm = torch.randperm(self.n, device=self.device)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        # Pre-compute offset vector once (reused every batch)
        _offsets = torch.arange(self.window_size, device=self.device)
        for start in range(0, self.n, self.batch_size):
            flat = self._perm[start:start + self.batch_size]
            tj = flat // self.wpt
            ws = flat % self.wpt
            # Vectorized gather — no Python loop, no GPU→CPU sync
            time_idx = ws.unsqueeze(1) + _offsets.unsqueeze(0)  # [B, W]
            windows = self.trajectories[tj.unsqueeze(1).expand_as(time_idx), time_idx]
            yield {
                'trajectories': windows,
                'tau': self.tau[tj],
                'alpha': self.alpha[tj],
                'rho_cp': self.rho_cp[tj],
                'q': self.q[tj],
                'bc_left': self.bc_left[tj],
                'bc_right': self.bc_right[tj],
                'dt': self.dt,
                'dx': self.dx,
            }


# Keep legacy Dataset/DataLoader classes for CPU fallback

class CattaneoDataset(Dataset):
    """Fallback CPU Dataset for pair data."""
    def __init__(self, data: Dict):
        self.tensor_keys = ['T_n', 'T_nm1', 'T_target', 'T_prev', 'T_prev2',
                           'q', 'tau', 'alpha', 'rho_cp', 'bc_left', 'bc_right']
        self.scalar_keys = ['dt', 'dx']
        self.data = data
        self.length = len(data['T_n'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        result = {k: self.data[k][idx] for k in self.tensor_keys}
        result['T'] = self.data['T_target'][idx]
        for k in self.scalar_keys:
            result[k] = self.data[k]
        return result


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ['dt', 'dx']:
            result[key] = batch[0][key]
        else:
            result[key] = torch.stack([sample[key] for sample in batch])
    return result


class TrajectoryDataset(Dataset):
    """Fallback CPU Dataset for trajectory windows."""
    def __init__(self, traj_data: Dict, window_size: int = 12):
        self.trajectories = traj_data['trajectories']
        self.tau = traj_data['tau']
        self.alpha = traj_data['alpha']
        self.rho_cp = traj_data['rho_cp']
        self.q = traj_data['q']
        self.bc_left = traj_data['bc_left']
        self.bc_right = traj_data['bc_right']
        self.dt = traj_data['dt']
        self.dx = traj_data['dx']
        self.window_size = window_size
        n_traj, seq_len, _ = self.trajectories.shape
        self.windows_per_traj = max(1, seq_len - window_size + 1)
        self.length = n_traj * self.windows_per_traj

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ti = idx // self.windows_per_traj
        ws = idx % self.windows_per_traj
        return {
            'trajectories': self.trajectories[ti, ws:ws+self.window_size],
            'tau': self.tau[ti], 'alpha': self.alpha[ti],
            'rho_cp': self.rho_cp[ti], 'q': self.q[ti],
            'bc_left': self.bc_left[ti], 'bc_right': self.bc_right[ti],
            'dt': self.dt, 'dx': self.dx,
        }


def trajectory_collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ['dt', 'dx']:
            result[key] = batch[0][key]
        else:
            result[key] = torch.stack([sample[key] for sample in batch])
    return result


def _impose_exact_dirichlet_bc(tensor: torch.Tensor,
                               bc_left: torch.Tensor,
                               bc_right: torch.Tensor) -> torch.Tensor:
    """Project tensor endpoints onto the exact Dirichlet BCs."""
    projected = tensor.clone()
    view_shape = [bc_left.shape[0]] + [1] * (projected.dim() - 1)
    left = bc_left.to(device=projected.device, dtype=projected.dtype).reshape(view_shape)
    right = bc_right.to(device=projected.device, dtype=projected.dtype).reshape(view_shape)
    projected[..., 0] = left.expand_as(projected[..., :1]).squeeze(-1)
    projected[..., -1] = right.expand_as(projected[..., :1]).squeeze(-1)
    return projected


def _project_dataset_to_exact_bc(data: Optional[Dict]) -> Optional[Dict]:
    """Align stored states with the model's hard Dirichlet boundary semantics."""
    if data is None:
        return None

    projected = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }
    bc_left = projected.get('bc_left')
    bc_right = projected.get('bc_right')
    if bc_left is None or bc_right is None:
        return projected

    for key in ('T_n', 'T_nm1', 'T_target', 'T_prev', 'T_prev2', 'trajectories'):
        value = projected.get(key)
        if isinstance(value, torch.Tensor):
            projected[key] = _impose_exact_dirichlet_bc(value, bc_left, bc_right)

    return projected


def _clip_extreme_samples(data: Optional[Dict], max_inc_dimless: float = 2.0) -> Optional[Dict]:
    """Remove pair samples whose increment RMSE exceeds a threshold.

    This prevents extreme GP-generated outliers from destabilising training.
    Threshold is in dimensionless T* units (multiply by delta_T for Kelvin).
    """
    if data is None:
        return None
    T_n = data.get('T_n')
    T_target = data.get('T_target')
    if T_n is None or T_target is None:
        return data  # trajectory-only dict

    inc = T_target - T_n  # [N, grid]
    per_sample_rmse = inc.pow(2).mean(dim=-1).sqrt()  # [N]
    keep = per_sample_rmse <= max_inc_dimless
    n_orig = T_n.shape[0]
    n_keep = int(keep.sum().item())
    if n_keep < n_orig:
        print(f"  Clipped {n_orig - n_keep}/{n_orig} extreme samples "
              f"(inc RMSE > {max_inc_dimless:.1f} T*)")
        return {
            key: value[keep] if isinstance(value, torch.Tensor) and value.shape[0] == n_orig else value
            for key, value in data.items()
        }
    return data


class CattaneoTrainer:
    """
    Trainer for Cattaneo-LNO model.

    Features:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Loss weight annealing
    """

    def __init__(self, model, config: Dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.total_params = sum(p.numel() for p in model.parameters())

        # Performance: enable TF32 + cuDNN auto-tuner on CUDA
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')  # TF32 on Ampere+

        # Extract physical scales from model's scaler to ensure consistency
        scaler = model.scaler
        L = scaler.L
        alpha_ref = scaler.alpha_ref
        T_ref = scaler.T_ref
        delta_T = scaler.delta_T

        # Loss function (use same physical scales as model)
        self.criterion = CattaneoPhysicsLoss(
            lambda_cattaneo=config.get('lambda_cattaneo', 1.0),
            lambda_energy=config.get('lambda_energy', 0.1),
            lambda_characteristic=config.get('lambda_characteristic', 0.1),
            lambda_bc=config.get('lambda_bc', 0.0),
            lambda_dTdt=config.get('lambda_dTdt', 0.0),
            data_loss_floor_k=config.get('data_loss_floor_k', 1e-3),
            lambda_gain=config.get('lambda_gain', 1.0),
            L=L,
            alpha_ref=alpha_ref,
            T_ref=T_ref,
            delta_T=delta_T,
            tau_ref=scaler.tau_ref
        )

        self.adaptive_weighting_enabled = config.get('adaptive_weighting.enabled', False)
        if self.adaptive_weighting_enabled:
            self.adaptive_weights = AdaptiveLossWeights(
                strategy=config.get('adaptive_weighting.strategy', 'magnitude'),
                warmup_epochs=config.get('adaptive_weighting.warmup_epochs', 10),
                temperature=config.get('adaptive_weighting.temperature', 1.0),
                ema_decay=config.get('adaptive_weighting.ema_decay', 0.9),
            ).to(device)
            self.update_frequency = config.get('adaptive_weighting.update_frequency', 10)
        else:
            self.adaptive_weights = None

        # Optimizer - use AdamW with slightly higher weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),  # Increased for better regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler - smooth cosine annealing without restarts
        # Restarts cause 100x LR jumps that spike rollout MSE via error compounding
        self.scheduler_type = config.get('scheduler', 'cosine')
        epochs = config.get('epochs', 200)
        if self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=config.get('lr', 1e-3) * 0.01  # Min LR = 1% of initial
            )
        elif self.scheduler_type == 'cosine_warm_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(10, epochs // 5),
                T_mult=2,
                eta_min=config.get('lr', 1e-3) * 0.01
            )
        elif self.scheduler_type == 'one_cycle':
            # OneCycleLR provides automatic warmup and annealing.
            # Created with placeholder steps_per_epoch; recreated in train()
            # once the actual loader length is known.
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.get('lr', 1e-3),
                epochs=epochs,
                steps_per_epoch=config.get('steps_per_epoch', 100),
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:
            # Fallback to ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )

        # Mixed precision: prefer bf16 (fp32 range, no overflow) over fp16
        if device == 'cuda' and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.scaler = None  # GradScaler not needed for bf16
        elif device == 'cuda':
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler()
        else:
            self.amp_dtype = None
            self.scaler = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'val_data_loss': [],
            'val_physics_loss': [],
            'val_rollout_mse': [],
            'val_increment_rmse_k': [],
            'val_increment_rel_error': [],
            'val_increment_gain_ratio': [],
            'val_rollout_rmse_k': [],
            'epoch_time_s': [],
            'epoch_elapsed_s': [],
        }

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self._steady_state_coord_cache = {}

        # Rollout training hyperparameters (exposed as instance attributes for
        # easy sweep-time override and test-time inspection)
        # Quadratic warmup: λ(e) = λ_max * (e/warmup)^2 for gentler ramp
        self.rollout_steps_min = config.get('rollout_steps_min', 3)
        self.rollout_steps_max = config.get('rollout_steps_max', config.get('rollout_steps', 15))
        self.rollout_warmup_epochs = config.get('rollout_warmup_epochs', 15)  # Extended warmup
        self.lambda_rollout = config.get('lambda_rollout', 2.0)  # Reduced target weight
        self.rollout_every_n_batches = max(1, int(config.get('rollout_every_n_batches', 1)))
        self.rollout_physics_weight = config.get('rollout_physics_weight', 0.0)
        # Disable teacher forcing to force model to learn from its own predictions
        self.scheduled_sampling_start = config.get('scheduled_sampling_start', 0)
        self.scheduled_sampling_end = config.get('scheduled_sampling_end', 0)  # No teacher forcing
        self.timestep_jump = config.get('timestep_jump', 1)
        self.steady_state_every_n_batches = max(1, int(config.get('steady_state_every_n_batches', 4)))
        self.validation_max_batches = max(0, int(config.get('validation_max_batches', 0)))
        self.train_eval_max_batches = max(0, int(config.get('train_eval_max_batches', 4)))
        self.val_rollout_every_n_epochs = max(1, int(config.get('val_rollout_every_n_epochs', 1)))
        self.val_rollout_max_batches = max(1, int(config.get('val_rollout_max_batches', 8)))

        # Anti-drift losses
        self.lambda_steady_state = config.get('lambda_steady_state', 0.1)
        self.lambda_contraction = config.get('lambda_contraction', 0.1)

    def _compute_steady_state_profile(self, bc_left, bc_right, grid_size, device):
        """Compute analytical steady-state temperature (linear between BCs)."""
        cache_key = (grid_size, str(device), bc_left.dtype)
        x_norm = self._steady_state_coord_cache.get(cache_key)
        if x_norm is None:
            x_norm = torch.linspace(0, 1, grid_size, device=device, dtype=bc_left.dtype)
            self._steady_state_coord_cache[cache_key] = x_norm
        # bc_left/bc_right: [B] → [B, grid]
        return bc_left.unsqueeze(-1) + (bc_right - bc_left).unsqueeze(-1) * x_norm

    def _compute_steady_state_loss(self, batch_tensors):
        """Fixed-point loss: model should predict zero change at equilibrium.

        Feeds the analytical steady state (linear profile between BCs) and
        penalises any predicted temperature increment or non-zero dTdt.
        One forward pass — no rollout needed.
        """
        dt = batch_tensors['dt']
        dx = batch_tensors['dx']
        bc_l = batch_tensors['bc_left']
        bc_r = batch_tensors['bc_right']
        grid = batch_tensors['T_n'].shape[-1]

        T_ss = self._compute_steady_state_profile(bc_l, bc_r, grid, self.device)
        q = torch.zeros_like(batch_tensors['q'])

        pred = self.model(
            T_ss, T_ss, q,
            batch_tensors['tau'], batch_tensors['alpha'], batch_tensors['rho_cp'],
            bc_l, bc_r, dt, dx
        )

        T_ref = self.model.scaler.T_ref
        delta_T = self.model.scaler.delta_T

        # Increment should be zero at steady state
        inc_star = (pred['T_pred'] - T_ss) / delta_T
        loss = (inc_star ** 2).mean()

        # NOTE: we intentionally do NOT penalise dTdt here.  The dTdt head
        # must remain free to predict large transient rates for non-Fourier
        # wave dynamics; a zero-target penalty would bias its weights toward
        # smaller outputs everywhere.  The δT penalty above is sufficient to
        # anchor the fixed point.

        return loss

    def _compute_gt_rollout_loss(self, traj_batch, rollout_steps, teacher_forcing_ratio=0.0):
        """
        Pushforward rollout loss (Brandstetter et al., 2022) with contraction.

        At each autoregressive step, the model receives its OWN previous
        predictions as inputs — but those inputs are **detached** so that
        gradients flow through only a single model application per step,
        not the entire K-step chain.  This eliminates the
        vanishing/exploding gradient problem of full BPTT through long
        rollouts and produces a clean, stable training signal.

        Additionally computes a **contraction loss**: the Euclidean distance
        from the model's prediction to the analytical steady state must
        decrease at every step.  This prevents long-horizon drift without
        requiring longer rollouts.

        Loss is computed in **dimensionless** space (T*) so it is on
        the same scale as the single-step data loss.
        """
        trajs = traj_batch['trajectories']  # [B, window, grid]
        B, total_len, grid = trajs.shape
        K = min(rollout_steps, total_len - 2)
        if K < 1:
            return torch.tensor(0.0, device=self.device)

        dt = traj_batch['dt']
        dx = traj_batch['dx']
        q = traj_batch['q']
        tau = traj_batch['tau']
        alpha = traj_batch['alpha']
        rho_cp = traj_batch['rho_cp']
        bc_l = traj_batch['bc_left']
        bc_r = traj_batch['bc_right']

        T_ref = self.model.scaler.T_ref
        delta_T = self.model.scaler.delta_T

        T_prev = trajs[:, 0]
        T_curr = trajs[:, 1]

        loss = torch.tensor(0.0, device=self.device)
        loss_contraction = torch.tensor(0.0, device=self.device)
        hidden_state = None  # recurrent memory (None when disabled)

        # Steady-state profile for contraction loss
        lambda_ctr = self.lambda_contraction
        if lambda_ctr > 0:
            T_ss = self._compute_steady_state_profile(bc_l, bc_r, grid, self.device)
            T_ss_star = ((T_ss - T_ref) / delta_T).to(dtype=trajs.dtype)
            # Energy of initial input w.r.t. steady state
            T_curr_star_init = ((T_curr - T_ref) / delta_T).to(dtype=trajs.dtype)
            E_prev = ((T_curr_star_init - T_ss_star) ** 2).mean(dim=-1)  # [B]

        for k in range(K):
            # ── Pushforward: detach autoregressive inputs ──
            pred = self.model(
                T_curr.detach(), T_prev.detach(), q,
                tau, alpha, rho_cp,
                bc_l, bc_r, dt, dx,
                hidden_state=hidden_state.detach() if hidden_state is not None else None,
            )
            T_pred = pred['T_pred']
            hidden_state = pred.get('hidden_state')  # carry forward
            T_gt = trajs[:, k + 2]

            # Dimensionless MSE — same scale as single-step data loss
            T_pred_star = (T_pred - T_ref) / delta_T
            T_gt_star = ((T_gt - T_ref) / delta_T).to(dtype=T_pred_star.dtype)
            step_mse = F.mse_loss(T_pred_star, T_gt_star)
            # Clamp per-step MSE: on untrained models autoregressive error
            # compounds exponentially, producing losses of 1e+29 that hijack
            # all gradients away from the data loss.  Cap at 1e4 (≈ T* error
            # of 100, i.e. 10 000 K) — beyond this the gradient is noise.
            loss = loss + step_mse.clamp(max=1e4)

            # ── Contraction loss: penalise if prediction moves AWAY from T_ss ──
            if lambda_ctr > 0:
                E_curr = ((T_pred_star - T_ss_star) ** 2).mean(dim=-1)  # [B]
                # Hinge: only penalise when energy increases
                violation = F.relu(E_curr - E_prev)
                loss_contraction = loss_contraction + violation.mean()
                E_prev = E_curr.detach()  # detach to avoid BPTT across steps

            # ── Scheduled sampling: mix GT and model prediction ──
            # teacher_forcing_ratio=1 → always use GT (teacher forcing)
            # teacher_forcing_ratio=0 → always use model prediction (pushforward)
            T_prev = T_curr
            if teacher_forcing_ratio > 0 and k + 2 < total_len:
                use_gt = random.random() < teacher_forcing_ratio
                T_curr = trajs[:, k + 2] if use_gt else T_pred
            else:
                T_curr = T_pred

        return loss / K + lambda_ctr * loss_contraction / K

    def train_epoch(self, train_loader: DataLoader,
                    traj_loader=None) -> Dict[str, float]:
        """Train for one epoch with gradient-balanced physics weighting."""
        self.model.train()

        grad_accum_steps = self.config.get('grad_accum_steps', 1)
        grad_balance_every = self.config.get('grad_balance_every', 5)
        # Accumulate on GPU to avoid per-batch CUDA syncs
        total_loss_t = torch.tensor(0.0, device=self.device)
        total_data_t = torch.tensor(0.0, device=self.device)
        total_physics_t = torch.tensor(0.0, device=self.device)
        total_rollout_t = torch.tensor(0.0, device=self.device)
        num_batches = 0

        # EMA of gradient-balanced weight (λ_phys)
        if not hasattr(self, '_ema_lambda_phys'):
            self._ema_lambda_phys = 1.0

        # Set up trajectory iterator if available
        traj_iter = iter(traj_loader) if traj_loader is not None else None

        pbar = tqdm(train_loader, desc='Training', mininterval=2.0)
        for batch_idx, batch in enumerate(pbar):
            # Data is already on the correct device (pre-transferred in Dataset)
            batch_tensors = batch

            # Extract scalars
            dt = batch_tensors['dt']
            dx = batch_tensors['dx']

            use_amp = self.amp_dtype is not None
            amp_ctx = torch.amp.autocast(self.device, dtype=self.amp_dtype) if use_amp else torch.enable_grad()

            with amp_ctx:
                # Input noise regularization: add small Gaussian noise to
                # T_n/T_nm1 during training so the model learns to correct
                # for its own prediction errors during autoregressive rollout.
                T_n_input = batch_tensors['T_n']
                T_nm1_input = batch_tensors['T_nm1']
                input_noise_std = self.config.get('input_noise_std', 0.0)
                if input_noise_std > 0 and self.model.training:
                    # Scale noise by delta_T to keep it in dimensionless units
                    noise_scale = input_noise_std * self.model.scaler.delta_T
                    T_n_input = T_n_input + torch.randn_like(T_n_input) * noise_scale
                    T_nm1_input = T_nm1_input + torch.randn_like(T_nm1_input) * noise_scale

                # Forward pass
                predictions = self.model(
                    T_n_input, T_nm1_input, batch_tensors['q'],
                    batch_tensors['tau'], batch_tensors['alpha'], batch_tensors['rho_cp'],
                    batch_tensors['bc_left'], batch_tensors['bc_right'],
                    dt, dx
                )

                # Adaptive weighting: call criterion once with current weights,
                # then update the weights from the individual raw-loss values.
                if hasattr(self, 'adaptive_weights') and self.adaptive_weights is not None:
                    losses = self.criterion(predictions, batch_tensors,
                                            adaptive_weights=self.adaptive_weights.get_weights())
                    current_losses = torch.stack([
                        losses['loss_data'], losses['loss_cattaneo'],
                        losses['loss_energy'], losses['loss_characteristic'],
                        losses['loss_bc']
                    ])
                    if getattr(self, 'current_epoch', 0) >= self.adaptive_weights.warmup_epochs and batch_idx % self.update_frequency == 0:
                        self.adaptive_weights.update(current_losses)
                else:
                    losses = self.criterion(predictions, batch_tensors)

                # ── Gradient-balanced physics weighting ──
                # Use loss-ratio proxy: λ_phys ≈ L_data / L_phys
                # Much cheaper than autograd.grad (no extra backward pass)
                # and well-correlated with the true gradient-norm ratio.
                loss_data = losses['loss_data']
                loss_physics = losses['loss_physics']
                pwf = self.criterion.physics_warmup_factor

                if (pwf > 0 and batch_idx % grad_balance_every == 0):
                    ld = loss_data.detach().item()
                    lp = loss_physics.detach().item()
                    if lp > 1e-12:
                        lam = ld / lp
                        lam = min(max(lam, 0.01), 100.0)
                        self._ema_lambda_phys = (
                            0.9 * self._ema_lambda_phys + 0.1 * lam)

                lambda_phys = self._ema_lambda_phys
                loss = loss_data + pwf * lambda_phys * loss_physics
                if losses.get('loss_bc', 0) > 0:
                    loss = loss + self.criterion.lambda_bc * losses['loss_bc']
                # Coefficient regularization: keep aθ, bθ near analytical targets
                loss_coeff_reg = losses.get('loss_coeff_reg', 0)
                if isinstance(loss_coeff_reg, torch.Tensor) and loss_coeff_reg.item() > 0:
                    loss = loss + 0.05 * loss_coeff_reg

                # Ground-truth rollout loss (pushforward, every batch)
                loss_rollout_val = 0.0
                should_compute_rollout = (
                    traj_iter is not None and
                    getattr(self, 'current_lambda_rollout', 0) > 0 and
                    batch_idx % self.rollout_every_n_batches == 0
                )
                if should_compute_rollout:
                    try:
                        traj_batch = next(traj_iter)
                    except StopIteration:
                        traj_iter = iter(traj_loader)
                        traj_batch = next(traj_iter)
                    rollout_steps = getattr(self, 'current_rollout_steps', 10)
                    tf_ratio = getattr(self, 'current_tf_ratio', 0.0)
                    loss_rollout = self._compute_gt_rollout_loss(
                        traj_batch, rollout_steps, teacher_forcing_ratio=tf_ratio)
                    loss = loss + self.current_lambda_rollout * loss_rollout
                    loss_rollout_val = loss_rollout.detach()

                # Autoencoder reconstruction loss (if model is wrapped)
                if 'loss_recon' in predictions and predictions['loss_recon'] is not None:
                    loss = loss + predictions['loss_recon']

            # Backward pass (gradient-accumulation aware)
            if batch_idx % grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            loss_scaled = loss / grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            is_update_step = (
                (batch_idx + 1) % grad_accum_steps == 0 or
                (batch_idx + 1) == len(train_loader)
            )
            if is_update_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Gradient noise: inject small Gaussian perturbations into the
                # accumulated gradients when a rollout loss was active.  This
                # regularises BPTT by preventing over-fitting to the exact
                # gradient direction of the unrolled trajectory.
                _gnoise = self.config.get('gradient_noise_std', 0.0)
                if _gnoise > 0.0 and should_compute_rollout:
                    for _p in self.model.parameters():
                        if _p.grad is not None:
                            _p.grad.add_(torch.randn_like(_p.grad) * _gnoise)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # OneCycleLR must be stepped every optimizer step
                if self.scheduler_type == 'one_cycle':
                    self.scheduler.step()

            # Accumulate losses on GPU (no sync)
            total_loss_t += loss.detach()
            total_data_t += losses['loss_data'].detach()
            total_physics_t += (losses['loss_cattaneo'] + losses['loss_energy']
                                + losses['loss_characteristic']).detach()
            total_rollout_t += loss_rollout_val if isinstance(loss_rollout_val, torch.Tensor) else 0.0
            num_batches += 1

            # Update progress bar every 10 batches to reduce sync overhead
            if batch_idx % 10 == 0:
                _disp = loss.item()
                pbar.set_postfix({'loss': f"{_disp:.2e}"})

        # Single GPU→CPU sync at epoch end
        return {
            'loss': (total_loss_t / num_batches).item(),
            'data_loss': (total_data_t / num_batches).item(),
            'physics_loss': (total_physics_t / num_batches).item(),
            'rollout_loss': (total_rollout_t / num_batches).item() if isinstance(total_rollout_t, torch.Tensor) else 0.0,
        }

    def validate(self, val_loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Validate on validation set - optimized with inference_mode."""
        self.model.eval()

        total_loss_t = torch.tensor(0.0, device=self.device)
        total_data_loss_t = torch.tensor(0.0, device=self.device)
        total_physics_loss_t = torch.tensor(0.0, device=self.device)
        total_increment_mse_t = torch.tensor(0.0, device=self.device)
        total_increment_target_energy_t = torch.tensor(0.0, device=self.device)
        total_increment_pred_energy_t = torch.tensor(0.0, device=self.device)
        num_batches = 0

        batch_limit = self.validation_max_batches if max_batches is None else max(0, int(max_batches))

        # Use inference_mode + autocast for faster inference
        use_amp = self.amp_dtype is not None
        amp_ctx = torch.amp.autocast(self.device, dtype=self.amp_dtype) if use_amp else nullcontext()
        with torch.inference_mode(), amp_ctx:
            for batch in val_loader:  # Removed tqdm for speed
                if batch_limit > 0 and num_batches >= batch_limit:
                    break
                batch_tensors = batch

                # Extract scalars
                dt = batch_tensors['dt']
                dx = batch_tensors['dx']

                # Forward pass
                predictions = self.model(
                    batch_tensors['T_n'], batch_tensors['T_nm1'], batch_tensors['q'],
                    batch_tensors['tau'], batch_tensors['alpha'], batch_tensors['rho_cp'],
                    batch_tensors['bc_left'], batch_tensors['bc_right'],
                    dt, dx
                )

                losses = self.criterion(predictions, batch_tensors)

                pred_increment = predictions['T_pred'] - batch_tensors['T_n']
                true_increment = batch_tensors['T'] - batch_tensors['T_n']
                increment_error = pred_increment - true_increment

                total_loss_t += losses['loss_total']
                total_data_loss_t += losses['loss_data']
                total_physics_loss_t += (losses['loss_cattaneo'] +
                                         losses['loss_energy'] +
                                         losses['loss_characteristic'])
                total_increment_mse_t += increment_error.pow(2).mean()
                total_increment_target_energy_t += true_increment.pow(2).mean()
                total_increment_pred_energy_t += pred_increment.pow(2).mean()
                num_batches += 1

        increment_mse = total_increment_mse_t / num_batches
        increment_target_energy = (total_increment_target_energy_t / num_batches).clamp(min=1e-30)
        increment_pred_energy = (total_increment_pred_energy_t / num_batches).clamp(min=1e-30)

        return {
            'loss': (total_loss_t / num_batches).item(),
            'data_loss': (total_data_loss_t / num_batches).item(),
            'physics_loss': (total_physics_loss_t / num_batches).item(),
            'increment_rmse_k': increment_mse.sqrt().item(),
            'increment_rel_error': torch.sqrt(increment_mse / increment_target_energy).item(),
            'increment_gain_ratio': torch.sqrt(increment_pred_energy / increment_target_energy).item(),
            'target_increment_rmse_k': increment_target_energy.sqrt().item(),
            'pred_increment_rmse_k': increment_pred_energy.sqrt().item(),
        }

    def validate_rollout(self, val_traj_loader, rollout_steps: int, max_batches: int = 8) -> Dict[str, float]:
        """Validate autoregressive rollout against ground-truth trajectories.
        
        Uses dimensionless T* space (same as training loss) so the metric
        is on the same scale as train/rollout_loss and directly comparable.
        Always called with a fixed K (rollout_steps_max) so the metric
        is not confounded by the warmup schedule.
        """
        self.model.eval()
        T_ref = self.model.scaler.T_ref
        delta_T = self.model.scaler.delta_T
        total_mse_t = torch.tensor(0.0, device=self.device)
        # Only track subset of steps for per-step MSE
        tracked_steps = list(range(0, rollout_steps, 3))  # Every 3rd step
        per_step_mse = {k: torch.tensor(0.0, device=self.device) for k in tracked_steps}
        num_batches = 0

        use_amp = self.amp_dtype is not None
        amp_ctx = torch.amp.autocast(self.device, dtype=self.amp_dtype) if use_amp else nullcontext()
        with torch.inference_mode(), amp_ctx:
            for traj_batch in val_traj_loader:
                if num_batches >= max_batches:
                    break  # Limit batches for speed
                    
                trajs = traj_batch['trajectories']
                B, total_len, grid = trajs.shape
                K = min(rollout_steps, total_len - 2)
                if K < 1:
                    continue

                T_prev = trajs[:, 0]
                T_curr = trajs[:, 1]

                batch_mse = torch.tensor(0.0, device=self.device)
                hidden_state = None  # recurrent memory (None when disabled)
                for k in range(K):
                    pred = self.model(
                        T_curr, T_prev, traj_batch['q'],
                        traj_batch['tau'], traj_batch['alpha'], traj_batch['rho_cp'],
                        traj_batch['bc_left'], traj_batch['bc_right'],
                        traj_batch['dt'], traj_batch['dx'],
                        hidden_state=hidden_state,
                    )
                    T_pred = pred['T_pred']
                    hidden_state = pred.get('hidden_state')  # carry forward
                    T_gt = trajs[:, k + 2]

                    # Dimensionless MSE — same scale as training rollout loss
                    T_pred_star = (T_pred - T_ref) / delta_T
                    T_gt_star = ((T_gt - T_ref) / delta_T).to(dtype=T_pred_star.dtype)
                    step_mse = F.mse_loss(T_pred_star, T_gt_star)
                    if k in per_step_mse:
                        per_step_mse[k] += step_mse
                    batch_mse += step_mse

                    # Autoregressive: feed model's own prediction
                    T_prev = T_curr
                    T_curr = T_pred

                total_mse_t += (batch_mse / K)
                num_batches += 1

        if num_batches == 0:
            return {'rollout_mse': 0.0}

        rollout_mse = (total_mse_t / num_batches).item()
        result = {
            'rollout_mse': rollout_mse,
            'rollout_rmse_k': math.sqrt(max(rollout_mse, 0.0)) * delta_T,
        }
        for k in tracked_steps:
            if k < rollout_steps:
                result[f'step_{k+1}_mse'] = (per_step_mse[k] / num_batches).item()
        return result

    def train(self, train_data: Dict, val_data: Dict,
             epochs: int = 200, batch_size: int = 32,
             checkpoint_dir: str = './checkpoints',
             resume_from: Optional[str] = None,
             wandb_log: bool = False,
             train_trajectories: Optional[Dict] = None,
             val_trajectories: Optional[Dict] = None):
        """
        Train the model.

        Args:
            train_data: Training dataset (pairs)
            val_data: Validation dataset (pairs)
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_dir: Directory to save checkpoints
            resume_from: Optional path to checkpoint to resume from
            wandb_log: Whether to log metrics to WandB
            train_trajectories: Optional trajectory data for rollout loss
            val_trajectories: Optional trajectory data for validation rollout
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_data = _project_dataset_to_exact_bc(train_data)
        val_data = _project_dataset_to_exact_bc(val_data)
        train_trajectories = _project_dataset_to_exact_bc(train_trajectories)
        val_trajectories = _project_dataset_to_exact_bc(val_trajectories)

        # Remove extreme outlier samples that destabilise training
        train_data = _clip_extreme_samples(train_data)
        val_data = _clip_extreme_samples(val_data)

        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_val_loss = checkpoint.get('val_data_loss', float('inf'))
            print(f"  Resumed at epoch {start_epoch}, best_val_loss(data)={self.best_val_loss:.4e}")

        # Build batch loaders — GPU-native when CUDA, DataLoader fallback on CPU
        _use_cuda = str(self.device).startswith('cuda')
        if _use_cuda:
            train_loader = GPUBatchLoader(train_data, batch_size, device=self.device, shuffle=True)
            val_loader = GPUBatchLoader(val_data, batch_size, device=self.device, shuffle=False)
        else:
            _nw = self.config.get('num_workers', 4)
            _persist = _nw > 0
            train_loader = DataLoader(
                CattaneoDataset(train_data), batch_size=batch_size,
                shuffle=True, num_workers=_nw, pin_memory=True,
                persistent_workers=_persist, prefetch_factor=2 if _persist else None,
                collate_fn=collate_fn)
            val_loader = DataLoader(
                CattaneoDataset(val_data), batch_size=batch_size,
                shuffle=False, num_workers=_nw, pin_memory=True,
                persistent_workers=_persist, prefetch_factor=2 if _persist else None,
                collate_fn=collate_fn)

        # Trajectory data loaders (for GT rollout loss)
        traj_loader = None
        val_traj_loader = None
        rollout_steps_max = self.rollout_steps_max
        rollout_steps_min = self.rollout_steps_min
        rollout_warmup_epochs = self.rollout_warmup_epochs
        lambda_rollout_target = self.lambda_rollout
        scheduled_sampling_start = self.scheduled_sampling_start
        scheduled_sampling_end = self.scheduled_sampling_end
        window_size = rollout_steps_max + 2  # need K+2 states

        _traj_bs = max(1, batch_size // 4)
        if train_trajectories is not None:
            print(f"Trajectory data available: {train_trajectories['trajectories'].shape[0]} trajectories")
            if _use_cuda:
                traj_loader = GPUTrajectoryLoader(
                    train_trajectories, window_size, _traj_bs, device=self.device, shuffle=True)
            else:
                traj_loader = DataLoader(
                    TrajectoryDataset(train_trajectories, window_size=window_size),
                    batch_size=_traj_bs, shuffle=True, num_workers=_nw, pin_memory=True,
                    persistent_workers=_persist, prefetch_factor=2 if _persist else None,
                    collate_fn=trajectory_collate_fn)
        if val_trajectories is not None:
            if _use_cuda:
                val_traj_loader = GPUTrajectoryLoader(
                    val_trajectories, window_size, _traj_bs, device=self.device, shuffle=False)
            else:
                val_traj_loader = DataLoader(
                    TrajectoryDataset(val_trajectories, window_size=window_size),
                    batch_size=_traj_bs, shuffle=False, num_workers=_nw, pin_memory=True,
                    persistent_workers=_persist, prefetch_factor=2 if _persist else None,
                    collate_fn=trajectory_collate_fn)

        print(f"Training on {train_loader.n if hasattr(train_loader, 'n') else len(train_loader.dataset)} samples")
        print(f"Validating on {val_loader.n if hasattr(val_loader, 'n') else len(val_loader.dataset)} samples")

        # Store the actual training dt from the data (may differ from CLI --dt)
        actual_dt = getattr(train_loader, 'dt', train_data.get('dt')) if isinstance(train_data, dict) else None
        if actual_dt is not None:
            self.config['training_dt'] = actual_dt
        if traj_loader is not None:
            print(f"Rollout training: K={rollout_steps_min}->{rollout_steps_max} over {rollout_warmup_epochs} epochs, "
                  f"lambda_rollout=0->{lambda_rollout_target}")
            print(f"Rollout loss computed every {self.rollout_every_n_batches} batch(es)")
        print(f"Steady-state loss computed every {self.steady_state_every_n_batches} batch(es)")
        if self.validation_max_batches > 0:
            print(f"Single-step validation capped at {self.validation_max_batches} batch(es)")
        if val_traj_loader is not None:
            print(f"Rollout validation every {self.val_rollout_every_n_epochs} epoch(s), "
                  f"max_batches={self.val_rollout_max_batches}")

        # Recreate OneCycleLR with the actual steps_per_epoch now that
        # the loader exists.  The placeholder created in __init__ has
        # the wrong total-step count, which corrupts the LR schedule.
        if self.scheduler_type == 'one_cycle':
            actual_steps = len(train_loader)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('lr', 1e-3),
                epochs=epochs,
                steps_per_epoch=actual_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            print(f"OneCycleLR: {actual_steps} steps/epoch, {actual_steps * epochs} total steps")

        training_start_time = time.perf_counter()

        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch
            total_epochs = start_epoch + epochs
            epoch_start_time = time.perf_counter()

            # ── Rollout schedule (quadratic warmup for gentler ramp) ──
            if traj_loader is not None and rollout_warmup_epochs > 0:
                linear_frac = min(1.0, epoch / rollout_warmup_epochs)
                # Quadratic warmup: slower at start, faster at end
                warmup_frac = linear_frac ** 2
            else:
                warmup_frac = 1.0
            self.current_rollout_steps = int(rollout_steps_min + warmup_frac * (rollout_steps_max - rollout_steps_min))
            # Lambda also uses quadratic ramp for gentler introduction
            self.current_lambda_rollout = warmup_frac * lambda_rollout_target if traj_loader is not None else 0.0
            # Store warmup_frac for data-loss annealing inside train_epoch
            self._current_warmup_frac = warmup_frac if traj_loader is not None else 0.0

            # ── Scheduled sampling (teacher forcing decay) ──
            if epoch < scheduled_sampling_start:
                self.current_tf_ratio = 1.0
            elif epoch >= scheduled_sampling_end:
                self.current_tf_ratio = 0.0
            else:
                self.current_tf_ratio = 1.0 - (epoch - scheduled_sampling_start) / max(1, scheduled_sampling_end - scheduled_sampling_start)

            # ── Physics curriculum schedule ──
            # Phase 1 (first 20%): data only (pwf = 0)
            # Phase 2 (next 30%):  mixed, ramp physics 0 → 1
            # Phase 3 (final 50%): physics-dominated (pwf = 1)
            physics_warmup_epochs = self.config.get('physics_warmup_epochs', 10)
            data_only_frac = self.config.get('curriculum_data_only_frac', 0.20)
            mixed_frac = self.config.get('curriculum_mixed_frac', 0.30)
            data_only_end = int(data_only_frac * total_epochs)
            mixed_end = int((data_only_frac + mixed_frac) * total_epochs)
            # Override physics_warmup_epochs if curriculum is configured
            if data_only_end > 0 or mixed_end > data_only_end:
                if epoch < data_only_end:
                    pwf = 0.0
                elif epoch < mixed_end:
                    pwf = (epoch - data_only_end) / max(1, mixed_end - data_only_end)
                else:
                    pwf = 1.0
            elif physics_warmup_epochs > 0 and epoch < physics_warmup_epochs:
                pwf = epoch / physics_warmup_epochs
            else:
                pwf = 1.0
            self.criterion.set_physics_warmup(pwf)

            # Reset best metric once rollout warmup completes so the model
            # can save new bests under the new composite metric.
            if (traj_loader is not None and rollout_warmup_epochs > 0
                    and epoch == rollout_warmup_epochs
                    and not getattr(self, '_rollout_metric_reset', False)):
                self._rollout_metric_reset = True
                self.best_val_loss = float('inf')
                self.epochs_without_improvement = 0
                print("  [INFO] Rollout warmup complete - resetting best metric and patience counter")

            print(f"\nEpoch {epoch + 1}/{total_epochs}  (physics_warmup={pwf:.2f}, "
                  f"rollout_K={self.current_rollout_steps}, lam_roll={self.current_lambda_rollout:.4f}, "
                  f"tf_ratio={self.current_tf_ratio:.2f})")

            # Shuffle GPU loaders at epoch start
            if hasattr(train_loader, 'shuffle'):
                train_loader.shuffle()
            if traj_loader is not None and hasattr(traj_loader, 'shuffle'):
                traj_loader.shuffle()

            # Train
            train_metrics = self.train_epoch(train_loader, traj_loader=traj_loader)

            # Validate (fast single-step validation)
            val_metrics = self.validate(val_loader)
            train_eval_metrics = None
            if self.train_eval_max_batches > 0:
                train_eval_metrics = self.validate(
                    train_loader,
                    max_batches=self.train_eval_max_batches,
                )

            # Validate rollout every epoch so composite metric is always consistent
            val_rollout_metrics = {}
            should_run_val_rollout = (
                val_traj_loader is not None and (
                    epoch == start_epoch or
                    (epoch + 1) % self.val_rollout_every_n_epochs == 0 or
                    (epoch + 1) == total_epochs
                )
            )
            if should_run_val_rollout:
                val_rollout_metrics = self.validate_rollout(
                    val_traj_loader,
                    rollout_steps_max,
                    max_batches=self.val_rollout_max_batches
                )
                self._latest_val_rollout_metrics = val_rollout_metrics
            elif hasattr(self, '_latest_val_rollout_metrics'):
                val_rollout_metrics = self._latest_val_rollout_metrics

            # Update scheduler (different handling for different scheduler types)
            if self.scheduler_type in ('cosine', 'cosine_warm_restarts'):
                self.scheduler.step()  # Epoch-based schedulers — no metric arg
            elif self.scheduler_type == 'one_cycle':
                pass  # OneCycleLR is stepped per batch, not per epoch
            else:
                self.scheduler.step(val_metrics['loss'])  # ReduceLROnPlateau needs metric

            # Store history
            epoch_duration_s = time.perf_counter() - epoch_start_time
            epoch_elapsed_s = time.perf_counter() - training_start_time
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_data_loss'].append(train_metrics['data_loss'])
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['val_data_loss'].append(val_metrics['data_loss'])
            self.history['val_physics_loss'].append(val_metrics['physics_loss'])
            self.history['val_increment_rmse_k'].append(val_metrics['increment_rmse_k'])
            self.history['val_increment_rel_error'].append(val_metrics['increment_rel_error'])
            self.history['val_increment_gain_ratio'].append(val_metrics['increment_gain_ratio'])
            self.history['val_rollout_mse'].append(
                val_rollout_metrics.get('rollout_mse') if val_rollout_metrics else None
            )
            self.history['val_rollout_rmse_k'].append(
                val_rollout_metrics.get('rollout_rmse_k') if val_rollout_metrics else None
            )
            self.history['epoch_time_s'].append(epoch_duration_s)
            self.history['epoch_elapsed_s'].append(epoch_elapsed_s)

            # Print metrics
            print(f"Train Opt Loss: {train_metrics['loss']:.4e} "
                  f"(Data: {train_metrics['data_loss']:.4e}, "
                  f"Phys: {train_metrics['physics_loss']:.4e}, "
                f"Roll: {train_metrics.get('rollout_loss', 0):.4e}, "
                f"lambda_phys: {self._ema_lambda_phys:.2e})")
            if train_eval_metrics is not None:
                print(f"Train Eval: {train_eval_metrics['loss']:.4e} "
                    f"(Data: {train_eval_metrics['data_loss']:.4e}, "
                    f"Phys: {train_eval_metrics['physics_loss']:.4e})")
            print(f"Val Loss: {val_metrics['loss']:.4e} "
                  f"(Data: {val_metrics['data_loss']:.4e}, "
                  f"Phys: {val_metrics['physics_loss']:.4e})")
            print(f"Val Increment: rmse={val_metrics['increment_rmse_k']:.4e}K, "
                f"rel={val_metrics['increment_rel_error']:.4e}, "
                f"gain={val_metrics['increment_gain_ratio']:.4e}x")
            if val_rollout_metrics:
                print(f"Val Rollout: mse={val_rollout_metrics['rollout_mse']:.4e}, "
                    f"rmse={val_rollout_metrics['rollout_rmse_k']:.4e}K")
                # Print per-step MSE for first few steps
                steps_to_show = min(5, self.current_rollout_steps)
                step_strs = [f"s{k+1}={val_rollout_metrics.get(f'step_{k+1}_mse', 0):.2e}" for k in range(steps_to_show)]
                print(f"  Per-step: {', '.join(step_strs)}")

            # WandB logging
            if wandb_log:
                try:
                    import wandb
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/loss': train_metrics['loss'],
                        'train/data_loss': train_metrics['data_loss'],
                        'train/physics_loss': train_metrics['physics_loss'],
                        'train/rollout_loss': train_metrics.get('rollout_loss', 0),
                        'val/loss': val_metrics['loss'],
                        'val/data_loss': val_metrics['data_loss'],
                        'val/physics_loss': val_metrics['physics_loss'],
                        'val/increment_rmse_k': val_metrics['increment_rmse_k'],
                        'val/increment_rel_error': val_metrics['increment_rel_error'],
                        'val/increment_gain_ratio': val_metrics['increment_gain_ratio'],
                        'val/target_increment_rmse_k': val_metrics['target_increment_rmse_k'],
                        'val/pred_increment_rmse_k': val_metrics['pred_increment_rmse_k'],
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'schedule/rollout_steps': self.current_rollout_steps,
                        'schedule/lambda_rollout': self.current_lambda_rollout,
                        'schedule/tf_ratio': self.current_tf_ratio,
                    }
                    if val_rollout_metrics:
                        log_dict['val/rollout_mse'] = val_rollout_metrics['rollout_mse']
                        log_dict['val/rollout_rmse_k'] = val_rollout_metrics['rollout_rmse_k']
                        for k in range(self.current_rollout_steps):
                            key = f'step_{k+1}_mse'
                            if key in val_rollout_metrics:
                                log_dict[f'val_rollout/{key}'] = val_rollout_metrics[key]
                    # Model size
                    log_dict['model/total_params'] = self.total_params

                    # Combined efficiency metric: error × size penalty
                    # Penalises larger models — 4× params needs 2× lower MSE to tie
                    PARAM_REF = 100_000
                    PARAM_PENALTY = 0.5
                    rollout = log_dict.get('val/rollout_mse', log_dict['val/data_loss'])
                    size_factor = (self.total_params / PARAM_REF) ** PARAM_PENALTY
                    log_dict['val/efficiency_score'] = rollout * size_factor

                    if hasattr(self, 'adaptive_weights') and self.adaptive_weights is not None:
                        w = self.adaptive_weights.get_weights().detach().cpu().numpy()
                        log_dict.update({
                            'weights/data': w[0],
                            'weights/cattaneo': w[1],
                            'weights/energy': w[2],
                            'weights/characteristic': w[3],
                            'weights/bc': w[4]
                        })
                    wandb.log(log_dict)
                except Exception:
                    pass

            # Save best model using rollout fidelity as the primary signal when
            # trajectory validation is available. Single-step data loss alone
            # was selecting near-identity checkpoints that failed dynamic probes.
            checkpoint_metric = val_metrics['increment_rmse_k']
            rollout_metric = val_rollout_metrics.get('rollout_rmse_k') if val_rollout_metrics else None
            gain_penalty = abs(val_metrics['increment_gain_ratio'] - 1.0)
            if rollout_metric is not None:
                composite = checkpoint_metric + 0.25 * rollout_metric + 0.50 * gain_penalty
            else:
                composite = checkpoint_metric + 0.50 * gain_penalty
            if composite < self.best_val_loss:
                self.best_val_loss = composite
                self.epochs_without_improvement = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_data_loss': val_metrics['data_loss'],
                    'val_increment_rmse_k': val_metrics['increment_rmse_k'],
                    'val_increment_rel_error': val_metrics['increment_rel_error'],
                    'val_increment_gain_ratio': val_metrics['increment_gain_ratio'],
                    'val_rollout_mse': val_rollout_metrics.get('rollout_mse') if val_rollout_metrics else None,
                    'val_rollout_rmse_k': rollout_metric,
                    'config': self.config
                }

                torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
                if rollout_metric is not None:
                    print(f"Saved best model (inc_rmse={checkpoint_metric:.4e}K, rollout_rmse={rollout_metric:.4e}K, gain={val_metrics['increment_gain_ratio']:.4e}x, composite={composite:.4e})")
                else:
                    print(f"Saved best model (inc_rmse={checkpoint_metric:.4e}K, gain={val_metrics['increment_gain_ratio']:.4e}x, composite={composite:.4e})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.config.get('patience', 50):
                print(f"Early stopping after {epoch + 1} epochs")
                break

            # Save checkpoint periodically
            if (epoch + 1) % 50 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_data_loss': val_metrics['data_loss'],
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        # Save final model
        total_training_time_s = time.perf_counter() - training_start_time
        epochs_completed = len(self.history['train_loss'])
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config,
            'training_summary': {
                'framework': 'pytorch',
                'epochs_completed': epochs_completed,
                'total_training_time_s': total_training_time_s,
                'avg_epoch_time_s': total_training_time_s / max(epochs_completed, 1),
                'best_val_loss': self.best_val_loss,
            }
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'final_model.pt'))

        # Save history
        with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)

        with open(os.path.join(checkpoint_dir, 'training_summary.json'), 'w') as f:
            json.dump({
                'framework': 'pytorch',
                'epochs_completed': epochs_completed,
                'total_training_time_s': total_training_time_s,
                'avg_epoch_time_s': total_training_time_s / max(epochs_completed, 1),
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'epoch_time_s': self.history['epoch_time_s'],
                'epoch_elapsed_s': self.history['epoch_elapsed_s'],
            }, f, indent=2)

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4e}")

        return self.history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Cattaneo-LNO')
    parser.add_argument('--grid_size', type=int, default=112, help='Grid size')
    parser.add_argument('--dx', type=float, default=1e-8, help='Spatial step')
    parser.add_argument('--dt', type=float, default=1e-13, help='Time step')
    parser.add_argument('--n_train', type=int, default=1000, help='Training samples')
    parser.add_argument('--n_val', type=int, default=200, help='Validation samples')
    parser.add_argument('--timestep_jump', type=int, default=200, help='FDM steps per supervised pair')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--modes', type=int, default=16, help='Laplace poles')
    parser.add_argument('--width', type=int, default=64, help='Model width')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate training data
    print("Generating training data...")
    train_data, val_data, train_traj, val_traj = generate_training_data(
        grid_size=args.grid_size,
        dx=args.dx,
        dt=args.dt,
        n_train=args.n_train,
        n_val=args.n_val,
        timestep_jump=args.timestep_jump
    )

    # Create model
    print("Creating model...")
    L = args.grid_size * args.dx  # Domain size [m]
    model = create_cattaneo_model(
        grid_size=args.grid_size,
        L=L,              # Domain size [m]
        alpha_ref=1e-4,   # Reference diffusivity [m²/s]
        T_ref=200.0,      # Reference temperature [K]
        delta_T=100.0     # Temperature range [K]
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training config
    config = {
        'lr': args.lr,
        'weight_decay': 1e-4,  # Moderate regularization
        'lambda_cattaneo': 1.0,
        'lambda_energy': 0.1,
        'lambda_characteristic': 0.1,
        'lambda_bc': 0.0,
        'patience': 50
    }

    # Create trainer
    trainer = CattaneoTrainer(model, config, device)

    # Train
    history = trainer.train(
        train_data, val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        train_trajectories=train_traj,
        val_trajectories=val_traj,
    )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()