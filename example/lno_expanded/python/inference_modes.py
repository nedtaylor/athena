"""
Inference Modes for Cattaneo-LNO
================================

Provides three advanced inference classes:
  1. SuperResolutionInference  – predict on finer grids than training
  2. TemporalFrameGenerator    – generate dense temporal snapshots
  3. NeuralWarmStart           – bootstrap FDM with neural prediction

All classes accept a trained CattaneoLNO model and operate in
dimensional space externally, using the model's built-in
dimensionless scaler internally.

Performance notes:
  - Constant tensors (tau, alpha, rho_cp, BCs, q) are pre-allocated
    once per rollout and reused across steps.
  - State is kept as GPU tensors during rollout loops to avoid
    repeated numpy↔GPU round-trips.
  - torch.inference_mode() is used instead of torch.no_grad() for
    faster execution (skips version counting).
"""

import numpy as np
import torch
from typing import Optional, List, Dict
from scipy.interpolate import interp1d

from cattaneo_lno import CattaneoLNO


def _prealloc_buffers(grid_size: int, tau: float, alpha: float,
                      rho_cp: float, bc_left: float, bc_right: float,
                      device: str):
    """Pre-allocate constant tensors shared across rollout steps."""
    return {
        'q': torch.zeros(1, grid_size, device=device),
        'tau': torch.full((1, grid_size), tau, device=device),
        'alpha': torch.full((1, grid_size), alpha, device=device),
        'rho_cp': torch.full((1, grid_size), rho_cp, device=device),
        'bc_l': torch.tensor([bc_left], device=device),
        'bc_r': torch.tensor([bc_right], device=device),
    }


class SuperResolutionInference:
    """
    Spatial super-resolution: run the trained LNO on a finer grid
    than it was trained on.

    How it works:
    1. Interpolate the coarse-grid state to the fine grid.
    2. Run the LNO forward pass on the fine grid.
       The LNO's Laplace-domain convolutions naturally handle variable
       grid sizes because the kernel matrix is built from pairwise
       distances that adapt to the input length.
    3. Return the fine-grid prediction.

    The model must have been trained with `use_ghost_cells=True` so
    that the extended grid (fine_grid + 2) goes through the LNO.
    """

    def __init__(self, model: CattaneoLNO, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _interpolate_to_fine(self, field_coarse: np.ndarray,
                              coarse_size: int, fine_size: int) -> np.ndarray:
        """Cubic interpolation from coarse to fine grid."""
        x_coarse = np.linspace(0, 1, coarse_size)
        x_fine = np.linspace(0, 1, fine_size)
        f = interp1d(x_coarse, field_coarse, kind='cubic',
                     fill_value='extrapolate')
        return f(x_fine)

    @torch.inference_mode()
    def predict(self, T_coarse: np.ndarray, fine_grid_size: int,
                tau: float = 1e-9, alpha: float = 1e-4,
                rho_cp: float = 1e6,
                bc_left: float = 100.0, bc_right: float = 200.0,
                dt: float = 1e-13, dx: float = 1e-8,
                T_prev_coarse: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict one step on a fine grid given a coarse-grid state.

        Args:
            T_coarse: Temperature on coarse grid [coarse_size]
            fine_grid_size: Target fine grid size
            tau, alpha, rho_cp: Material properties (scalar, uniform)
            bc_left, bc_right: Boundary conditions [K]
            dt, dx: Time and spatial steps for coarse grid

        Returns:
            T_fine: Predicted temperature on fine grid [fine_size]
        """
        coarse_size = len(T_coarse)

        # Interpolate state to fine grid
        T_fine = self._interpolate_to_fine(T_coarse, coarse_size, fine_grid_size)

        # Interpolate previous state (or default to current)
        if T_prev_coarse is not None:
            T_prev_fine = self._interpolate_to_fine(T_prev_coarse, len(T_prev_coarse), fine_grid_size)
        else:
            T_prev_fine = T_fine.copy()

        # Fine-grid dx
        L = self.model.scaler.L
        dx_fine = L / fine_grid_size

        # Use the same macro time step as training.  The model's iterative
        # corrector adds corrections that are NOT scaled by dt_star, so
        # the output magnitude is effectively determined by Fo and the
        # learned dynamics — not by dt_eff.  Each neural step therefore
        # advances the same physical time regardless of grid resolution.
        dt_eff = dt * self.model.timestep_jump

        # Build tensors
        T_t = torch.from_numpy(T_fine).float().unsqueeze(0).to(self.device)
        T_prev_t = torch.from_numpy(T_prev_fine).float().unsqueeze(0).to(self.device)
        buf = _prealloc_buffers(fine_grid_size, tau, alpha, rho_cp,
                                bc_left, bc_right, self.device)

        # Forward pass on fine grid
        output = self.model(T_t, T_prev_t, buf['q'], buf['tau'], buf['alpha'],
                           buf['rho_cp'], buf['bc_l'], buf['bc_r'], dt_eff, dx_fine)

        T_pred = output['T_pred']
        # Re-enforce boundary conditions
        T_pred[:, 0] = bc_left
        T_pred[:, -1] = bc_right

        return T_pred.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def rollout(self, T_coarse: np.ndarray, fine_grid_size: int,
                num_steps: int = 10,
                tau: float = 1e-9, alpha: float = 1e-4,
                rho_cp: float = 1e6,
                bc_left: float = 100.0, bc_right: float = 200.0,
                dt: float = 1e-13, dx: float = 1e-8) -> List[np.ndarray]:
        """Multi-step rollout on fine grid, staying on GPU between steps.

        Each neural step advances the same physical time as on the
        training grid (dt * timestep_jump) because the model's iterative
        corrector is not scaled by dt_star.
        """
        coarse_size = len(T_coarse)
        T_np = self._interpolate_to_fine(T_coarse, coarse_size, fine_grid_size)
        T_t = torch.from_numpy(T_np).float().unsqueeze(0).to(self.device)
        T_prev_t = T_t.clone()
        history = [T_np.copy()]

        L = self.model.scaler.L
        dx_fine = L / fine_grid_size
        # Same macro time step as training — see predict() for rationale.
        dt_eff = dt * self.model.timestep_jump
        buf = _prealloc_buffers(fine_grid_size, tau, alpha, rho_cp,
                                bc_left, bc_right, self.device)

        hidden_state = None  # recurrent memory
        history_gpu = []
        for _ in range(num_steps):
            output = self.model(T_t, T_prev_t, buf['q'], buf['tau'],
                               buf['alpha'], buf['rho_cp'], buf['bc_l'],
                               buf['bc_r'], dt_eff, dx_fine,
                               hidden_state=hidden_state)
            T_next = output['T_pred']
            hidden_state = output.get('hidden_state')
            # Re-enforce boundary conditions
            T_next[:, 0] = bc_left
            T_next[:, -1] = bc_right
            T_prev_t = T_t
            T_t = T_next
            history_gpu.append(T_t.squeeze(0).detach())

        # Transfer to CPU once after the loop
        history = [T_np.copy()] + [t.cpu().numpy() for t in history_gpu]
        del history_gpu

        return history, dt_eff


class TemporalFrameGenerator:
    """
    Generate dense temporal frames via neural sub-stepping.

    The model is trained to predict `timestep_jump` FDM steps at once.
    By running it repeatedly with smaller effective dt (sub-stepping),
    we can generate temporally-denser snapshots.  Linear interpolation
    between neural steps provides even denser frames.
    """

    def __init__(self, model: CattaneoLNO, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, T0: np.ndarray, num_frames: int = 50,
                 sub_steps: int = 5,
                 tau: float = 1e-9, alpha: float = 1e-4,
                 rho_cp: float = 1e6,
                 bc_left: float = 100.0, bc_right: float = 200.0,
                 dt: float = 1e-13, dx: float = 1e-8) -> List[np.ndarray]:
        """Generate num_frames temporal snapshots."""
        grid_size = len(T0)
        frames = [T0.copy()]
        T_t = torch.from_numpy(T0).float().unsqueeze(0).to(self.device)
        T_prev_t = T_t.clone()

        dt_eff = dt * self.model.timestep_jump
        buf = _prealloc_buffers(grid_size, tau, alpha, rho_cp,
                                bc_left, bc_right, self.device)

        neural_steps = (num_frames + sub_steps - 1) // sub_steps

        # Keep numpy copies for interpolation
        T_np = T0.copy()

        hidden_state = None  # recurrent memory
        for i in range(neural_steps):
            output = self.model(T_t, T_prev_t, buf['q'], buf['tau'],
                               buf['alpha'], buf['rho_cp'], buf['bc_l'],
                               buf['bc_r'], dt_eff, dx,
                               hidden_state=hidden_state)
            T_next_t = output['T_pred']
            hidden_state = output.get('hidden_state')
            T_next_np = T_next_t.squeeze(0).cpu().numpy()

            remaining = num_frames - len(frames) + 1
            n_interp = min(sub_steps, remaining)
            for j in range(1, n_interp + 1):
                alpha_interp = j / sub_steps
                T_interp = (1 - alpha_interp) * T_np + alpha_interp * T_next_np
                frames.append(T_interp)

            T_prev_t = T_t
            T_t = T_next_t
            T_np = T_next_np

            if len(frames) > num_frames:
                break

        return frames[:num_frames + 1]


class NeuralWarmStart:
    """
    Neural warm-start for FDM: use the neural model to quickly advance
    the solution forward in time, then switch to FDM for high-accuracy
    refinement.
    """

    def __init__(self, model: CattaneoLNO, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def _neural_rollout(self, T0: np.ndarray, num_steps: int,
                        tau: float, alpha: float, rho_cp: float,
                        bc_left: float, bc_right: float,
                        dt: float, dx: float):
        """Fast GPU-resident neural rollout for phase 1."""
        grid_size = len(T0)
        T_t = torch.from_numpy(T0).float().unsqueeze(0).to(self.device)
        T_prev_t = T_t.clone()
        dt_eff = dt * self.model.timestep_jump
        buf = _prealloc_buffers(grid_size, tau, alpha, rho_cp,
                                bc_left, bc_right, self.device)

        hidden_state = None  # recurrent memory
        for _ in range(num_steps):
            output = self.model(T_t, T_prev_t, buf['q'], buf['tau'],
                               buf['alpha'], buf['rho_cp'], buf['bc_l'],
                               buf['bc_r'], dt_eff, dx,
                               hidden_state=hidden_state)
            T_next = output['T_pred']
            hidden_state = output.get('hidden_state')
            T_prev_t = T_t
            T_t = T_next

        T = T_t.squeeze(0).cpu().numpy()
        T_prev = T_prev_t.squeeze(0).cpu().numpy()
        return T, T_prev

    def solve(self, T0: np.ndarray, grid: np.ndarray,
              num_neural_steps: int = 50, num_fdm_steps: int = 500,
              tau: float = 1e-9, alpha: float = 1e-4,
              rho_cp: float = 1e6,
              bc_left: float = 100.0, bc_right: float = 200.0,
              dx: float = 1e-8, dt: float = 1e-13) -> dict:
        """
        Neural warm-start then FDM refinement.

        Phase 1: num_neural_steps fast neural forward passes on GPU.
        Phase 2: num_fdm_steps individual FDM sub-steps.
        """
        from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

        grid_size = len(T0)
        timestep_jump = self.model.timestep_jump

        # Phase 1: GPU-resident neural rollout
        print(f"  Phase 1: {num_neural_steps} neural steps "
              f"(replacing {num_neural_steps * timestep_jump} FDM sub-steps)...")
        T, T_prev = self._neural_rollout(
            T0, num_neural_steps, tau=tau, alpha=alpha, rho_cp=rho_cp,
            bc_left=bc_left, bc_right=bc_right, dt=dt, dx=dx)
        print(f"    After neural: T range [{T.min():.2f}, {T.max():.2f}]")

        # Phase 2: FDM refinement from the neural-advanced state
        # Each "macro step" = timestep_jump FDM sub-steps so that one
        # FDM refinement step covers the same physical time as one neural step.
        total_fdm_sub = num_fdm_steps * timestep_jump
        print(f"  Phase 2: {num_fdm_steps} macro steps "
              f"({total_fdm_sub} FDM sub-steps)...")
        for step in range(total_fdm_sub):
            T_new, _, _ = nl_solve_HF_1d_Cattaneo(
                grid_size, grid, T, T_prev,
                dx=dx, dt=dt, tau=tau,
                tol=1e-6, max_newton_iters=50,
                k_temp_dependent=False, c_temp_dependent=False,
                verbose=False,
                BC=(bc_left, bc_right)
            )
            T_prev = T
            T = T_new

        print(f"    After FDM: T range [{T.min():.2f}, {T.max():.2f}]")
        return {
            'T_final': T,
        }


if __name__ == '__main__':
    print("Inference modes loaded.")
    print("Available classes:")
    print("  - SuperResolutionInference")
    print("  - TemporalFrameGenerator")
    print("  - NeuralWarmStart")
