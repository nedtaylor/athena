"""
Data Generation for Cattaneo-LNO Training
==========================================

Generates training data via FDM-forward simulation:
1. Generate random temperature fields using Gaussian Processes
2. Run the FDM solver forward to produce genuine PDE evolution targets
3. Heat source q = 0 for all free-evolution data (no reverse-engineered sources)

All batches (GP, step-BC, equilibrium, trajectory) use q = 0, ensuring
no scaling mismatch between data generation and the physics-informed loss.

Material-property consistency:
    The FDM solver (nl_solve_HF_1d_Cattaneo) obtains its thermal
    conductivity k and heat capacity Cv from material data files
    (NHC_1.txt, NK_1.txt) — it does NOT accept user-supplied alpha or
    rho_cp.  With k_temp_dependent=False and c_temp_dependent=False
    the solver uses k(T_ref) and Cv(T_ref) from material file 1.

    Therefore alpha = k(T_ref)/Cv(T_ref) and rho_cp = Cv(T_ref) are
    DERIVED from the material file and recorded in the batch, NOT
    randomly sampled.  Only tau is randomly varied (it IS passed to
    the solver).

Nondimensionalization uses the Fourier number (Fo) and Vernotte number (Ve)
instead of the previous C = τ/t_c scaling.
"""

import numpy as np
import torch
import math
import os
import multiprocessing as mp
from functools import partial
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline
from typing import Tuple, Dict, List, Optional
import warnings

try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _HAS_CUPY = False

# Number of parallel workers for FDM sample generation.
# Leave one core free for the OS / main process.
_NUM_WORKERS = max(1, min(os.cpu_count() or 1, 8) - 1)


def _batched_tridiagonal_solve_gpu(lower, diag, upper, rhs):
    """Solve a batch of tridiagonal systems on GPU with shared off-diagonals."""
    batch_size, n = diag.shape
    if n == 1:
        return rhs / diag

    c_prime = cp.empty((batch_size, n - 1), dtype=diag.dtype)
    d_prime = cp.empty((batch_size, n), dtype=diag.dtype)

    beta = diag[:, 0]
    c_prime[:, 0] = upper[0] / beta
    d_prime[:, 0] = rhs[:, 0] / beta

    for i in range(1, n - 1):
        beta = diag[:, i] - lower[i - 1] * c_prime[:, i - 1]
        c_prime[:, i] = upper[i] / beta
        d_prime[:, i] = (rhs[:, i] - lower[i - 1] * d_prime[:, i - 1]) / beta

    beta = diag[:, -1] - lower[-1] * c_prime[:, -1]
    d_prime[:, -1] = (rhs[:, -1] - lower[-1] * d_prime[:, -2]) / beta

    x = cp.empty((batch_size, n), dtype=diag.dtype)
    x[:, -1] = d_prime[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[:, i] * x[:, i + 1]
    return x


def _fdm_trajectory_batch_gpu(T0_batch, tau_batch, dx, dt, timestep_jump,
                              bc_left, bc_right, total_jumps,
                              heat_capacity, conductivity):
    """Run a batch of constant-property 1D trajectories entirely on GPU."""
    if not _HAS_CUPY:
        raise RuntimeError("CuPy with CUDA support is required for GPU trajectory generation")

    T = cp.asarray(T0_batch, dtype=cp.float64)
    T_prev = T.copy()
    tau = cp.asarray(tau_batch, dtype=cp.float64)
    bc_left = cp.asarray(bc_left, dtype=cp.float64)
    bc_right = cp.asarray(bc_right, dtype=cp.float64)

    batch_size, grid_size = T.shape
    coeff = conductivity / (dx * dx)
    inv_dt = 1.0 / dt
    inv_dt2 = 1.0 / (dt * dt)

    lower = cp.full(grid_size - 1, -coeff, dtype=cp.float64)
    upper = cp.full(grid_size - 1, -coeff, dtype=cp.float64)
    diag_base = cp.full(grid_size, 2.0 * coeff, dtype=cp.float64)
    BA = cp.zeros((batch_size, grid_size), dtype=cp.float64)
    BA[:, 0] = coeff * bc_left
    BA[:, -1] = coeff * bc_right

    trajectories = cp.empty((batch_size, total_jumps + 1, grid_size), dtype=cp.float64)
    T[:, 0] = bc_left
    T[:, -1] = bc_right
    T_prev[:, 0] = bc_left
    T_prev[:, -1] = bc_right
    trajectories[:, 0, :] = T

    cv = cp.asarray(heat_capacity, dtype=cp.float64)
    if cv.ndim == 0:
        cv = cp.full((batch_size, grid_size), float(cv), dtype=cp.float64)
    elif cv.ndim == 1:
        cv = cp.broadcast_to(cv[None, :], (batch_size, grid_size))

    for jump_idx in range(total_jumps):
        for _ in range(timestep_jump):
            gamma = cv * (inv_dt + tau * inv_dt2)
            diag = diag_base[None, :] + gamma
            omega = -cv * T * inv_dt + tau * cv * (T_prev - 2.0 * T) * inv_dt2
            rhs = BA - omega
            T_new = _batched_tridiagonal_solve_gpu(lower, diag, upper, rhs)
            T_new[:, 0] = bc_left
            T_new[:, -1] = bc_right
            T_prev = T
            T = T_new
        trajectories[:, jump_idx + 1, :] = T

    return cp.asnumpy(trajectories)


# ── Module-level worker functions (picklable on Windows) ──────────────

def _apply_exact_bc_np(T, bc_l, bc_r):
    """Project array endpoints onto the exact Dirichlet boundary values."""
    T = np.array(T, copy=True)
    T[..., 0] = bc_l
    T[..., -1] = bc_r
    return T

def _fdm_evolve(T0, tau_1d, grid_size, dx, dt, timestep_jump, bc_l, bc_r):
    """Run FDM forward for *timestep_jump* steps from T0. Returns final T."""
    from HF_Cattaneo import nl_solve_HF_1d_Cattaneo
    grid = np.ones(grid_size, dtype=int)
    T = _apply_exact_bc_np(T0, bc_l, bc_r)
    T_prev = T.copy()
    for _ in range(timestep_jump):
        T_new, _, _ = nl_solve_HF_1d_Cattaneo(
            grid_size, grid, T, T_prev,
            dx=dx, dt=dt, tau=tau_1d,
            tol=1e-6, max_newton_iters=50,
            k_temp_dependent=False, c_temp_dependent=False,
            const=True, verbose=False,
            BC=(bc_l, bc_r),
        )
        T_new = _apply_exact_bc_np(T_new, bc_l, bc_r)
        T_prev = T.copy()
        T = T_new
    return T


def _fdm_evolve_pair(T0, tau_1d, grid_size, dx, dt, timestep_jump, bc_l, bc_r):
    """Run FDM forward for 2*timestep_jump steps from T0.

    Returns (T_mid, T_end, T_target) — three states each separated by
    timestep_jump FDM sub-steps.  Use as:
        T_nm1 = T_mid   (has non-trivial velocity history)
        T_n   = T_end
        T_target = T_target
    """
    from HF_Cattaneo import nl_solve_HF_1d_Cattaneo
    grid = np.ones(grid_size, dtype=int)
    T = _apply_exact_bc_np(T0, bc_l, bc_r)
    T_prev = T.copy()
    # Phase 1: 0 → timestep_jump  →  T_mid
    for _ in range(timestep_jump):
        T_new, _, _ = nl_solve_HF_1d_Cattaneo(
            grid_size, grid, T, T_prev,
            dx=dx, dt=dt, tau=tau_1d,
            tol=1e-6, max_newton_iters=50,
            k_temp_dependent=False, c_temp_dependent=False,
            const=True, verbose=False,
            BC=(bc_l, bc_r),
        )
        T_new = _apply_exact_bc_np(T_new, bc_l, bc_r)
        T_prev = T.copy()
        T = T_new
    T_mid = T.copy()
    # Phase 2: timestep_jump → 2*timestep_jump  →  T_end
    for _ in range(timestep_jump):
        T_new, _, _ = nl_solve_HF_1d_Cattaneo(
            grid_size, grid, T, T_prev,
            dx=dx, dt=dt, tau=tau_1d,
            tol=1e-6, max_newton_iters=50,
            k_temp_dependent=False, c_temp_dependent=False,
            const=True, verbose=False,
            BC=(bc_l, bc_r),
        )
        T_new = _apply_exact_bc_np(T_new, bc_l, bc_r)
        T_prev = T.copy()
        T = T_new
    T_end = T.copy()
    # Phase 3: 2*timestep_jump → 3*timestep_jump  →  T_target
    for _ in range(timestep_jump):
        T_new, _, _ = nl_solve_HF_1d_Cattaneo(
            grid_size, grid, T, T_prev,
            dx=dx, dt=dt, tau=tau_1d,
            tol=1e-6, max_newton_iters=50,
            k_temp_dependent=False, c_temp_dependent=False,
            const=True, verbose=False,
            BC=(bc_l, bc_r),
        )
        T_new = _apply_exact_bc_np(T_new, bc_l, bc_r)
        T_prev = T.copy()
        T = T_new
    T_target = T.copy()
    return T_mid, T_end, T_target


def _fdm_trajectory(T0, tau_1d, grid_size, dx, dt, timestep_jump,
                    bc_l, bc_r, total_jumps):
    """Run a full FDM trajectory returning states at every jump boundary."""
    from HF_Cattaneo import nl_solve_HF_1d_Cattaneo
    grid = np.ones(grid_size, dtype=int)
    T = _apply_exact_bc_np(T0, bc_l, bc_r)
    trajectory = [T.copy()]
    T_prev = T.copy()
    for _ in range(total_jumps):
        for _ in range(timestep_jump):
            T_new, _, _ = nl_solve_HF_1d_Cattaneo(
                grid_size, grid, T, T_prev,
                dx=dx, dt=dt, tau=tau_1d,
                tol=1e-6, max_newton_iters=50,
                k_temp_dependent=False, c_temp_dependent=False,
                const=True, verbose=False,
                BC=(bc_l, bc_r),
            )
            T_new = _apply_exact_bc_np(T_new, bc_l, bc_r)
            T_prev = T.copy()
            T = T_new
        trajectory.append(T.copy())
    return trajectory


class GaussianProcessSampler:
    """
    Gaussian Process sampler for generating smooth temperature fields.
    Uses RBF kernel with configurable length scale.
    """
    def __init__(self, grid_size: int, length_scale: float = 0.1, 
                 variance: float = 1.0, mean_temp: float = 200.0):
        self.grid_size = grid_size
        self.length_scale = length_scale
        self.variance = variance
        self.mean_temp = mean_temp
        
        # Create grid
        self.x = np.linspace(0, 1, grid_size)
        
        # Precompute covariance matrix
        self.K = self._compute_kernel_matrix()
        
        # Cholesky decomposition for sampling
        try:
            self.L = np.linalg.cholesky(self.K + 1e-6 * np.eye(grid_size))
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(self.K)
            eigvals = np.maximum(eigvals, 1e-6)
            self.L = eigvecs @ np.diag(np.sqrt(eigvals))
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        dists = cdist(x1.reshape(-1, 1), x2.reshape(-1, 1), 'sqeuclidean')
        return self.variance * np.exp(-0.5 * dists / (self.length_scale ** 2))
    
    def _compute_kernel_matrix(self) -> np.ndarray:
        """Compute covariance matrix for the grid."""
        return self._rbf_kernel(self.x, self.x)
    
    def sample(self, n_samples: int, temp_range: Tuple[float, float] = (10, 500)) -> np.ndarray:
        """
        Sample temperature fields from GP.
        
        Args:
            n_samples: Number of samples
            temp_range: (min, max) temperature range
            
        Returns:
            temperatures: [n_samples, grid_size]
        """
        # Sample from standard normal
        z = np.random.randn(n_samples, self.grid_size)
        
        # Transform through covariance
        samples = self.mean_temp + z @ self.L.T
        
        # Scale to desired range
        samples_min = samples.min(axis=1, keepdims=True)
        samples_max = samples.max(axis=1, keepdims=True)
        samples = temp_range[0] + (temp_range[1] - temp_range[0]) * (samples - samples_min) / (samples_max - samples_min + 1e-10)
        
        return samples


class CattaneoDataGenerator:
    """
    Data generator for Cattaneo equation using FDM-forward simulation.


    Generates training data via FDM forward stepping for all sample
    types, ensuring the model learns genuine PDE evolution.

    **Material-property consistency**:
        The FDM solver uses thermal conductivity k and heat capacity Cv
        from material data files, NOT user-supplied alpha/rho_cp.
        With k_temp_dependent=False and c_temp_dependent=False the
        solver uses k(T_ref) and Cv(T_ref) from material 1, giving a
        FIXED alpha = k/Cv across all grid points.

        Only tau is randomly varied per sample (and passed to the solver).
        alpha and rho_cp are derived from the material file so they match
        what the solver actually uses.

    **Fo/Ve nondimensionalization**:
        Fo = α · dt / dx²         (Fourier number — uniform, since α is fixed)
        Ve = √(α · τ) / L         (Vernotte number — varies with τ)
        T* = (T - T_ref) / ΔT
        dx* = dx / L

    The heat source q is set to zero for all free-evolution data.
    """
    def __init__(self, grid_size: int, dx: float, dt: float,
                 tau_range: Tuple[float, float] = (1e-10, 1e-8),
                 gp_length_scale: float = 0.1,
                 L: Optional[float] = None,
                 alpha_ref: float = 1e-4,
                 T_ref: float = 200.0,
                 delta_T: float = 100.0,
                 tau_ref: float = 1e-9,
                 material_file: int = 1,
                 fdm_backend: str = 'auto'):

        self.grid_size = grid_size
        self.dx = dx
        self.dt = dt
        self.tau_range = tau_range

        # Physical scales
        self.L = L if L is not None else grid_size * dx
        self.alpha_ref = alpha_ref
        self.T_ref = T_ref
        self.delta_T = delta_T
        self.tau_ref = tau_ref
        self.material_file = material_file
        self.fdm_backend = fdm_backend.lower()

        # ── Derive actual k and Cv from the material file at T_ref ──
        # This is what the FDM solver uses when k_temp_dependent=False
        # and c_temp_dependent=False.
        from HF_Cattaneo import _load_material
        material = _load_material(material_file)
        T_grid = material['T_grid']
        hc_values = material['hc_values']  # volumetric heat capacity
        tk_values = material['tk_values']  # thermal conductivity

        slope = (len(T_grid) - 1) / (material['T_max'] - material['T_min'])
        idx = int((T_ref - material['T_min']) * slope)
        idx = max(0, min(len(T_grid) - 2, idx))
        t = (T_ref - T_grid[idx]) / (T_grid[idx + 1] - T_grid[idx])

        self.Cv_mat = float(hc_values[idx] * (1 - t) + hc_values[idx + 1] * t)
        self.k_mat = float(tk_values[idx] * (1 - t) + tk_values[idx + 1] * t)
        self.alpha_mat = self.k_mat / self.Cv_mat   # actual α used by solver

        print(f"Material {material_file} at T_ref={T_ref}K: "
              f"k={self.k_mat:.4e}, Cv={self.Cv_mat:.4e}, "
              f"alpha=k/Cv={self.alpha_mat:.4e}")

        # Fo/Ve dimensionless groups (using the TRUE alpha from the material)
        self.Fo = max(1e-6, min(self.alpha_mat * dt / dx**2, 100.0))
        self.Ve = math.sqrt(self.alpha_mat * tau_ref) / self.L
        self.dx_star = dx / self.L

        # Diffusion timescale (for converting tau_range to dimensionless C)
        self.t_c = self.L**2 / self.alpha_mat

        # C = tau / t_c range (used to sample random tau values)
        self.C_range = (tau_range[0] / self.t_c, tau_range[1] / self.t_c)


        # GP sampler for dimensionless temperature fields T* in [-2, 3] (O(1))
        self.gp_sampler = GaussianProcessSampler(
            grid_size, gp_length_scale, variance=1.0, mean_temp=0.0
        )

    def _use_gpu_trajectory_backend(self, n_trajectories: int, total_jumps: int) -> bool:
        if self.fdm_backend == 'gpu':
            if not _HAS_CUPY:
                warnings.warn("fdm_backend='gpu' requested but CuPy is unavailable. Falling back to CPU.")
                return False
            return True
        if self.fdm_backend in ('cpu', 'newton'):
            return False
        return _HAS_CUPY and n_trajectories >= 16 and total_jumps >= 40

    def _sample_benchmark_relaxation_ic(self):
        """Benchmark-like zero-source relaxation IC.

        Uses a linear baseline with a dominant mode-1 sinusoidal hump and
        BCs centred near the benchmark 100K/200K setting so rollout training
        sees the same relaxation regime that dominates evaluation.
        """
        x_norm = np.linspace(0, 1, self.grid_size)
        bc_l = np.random.normal(100.0, 12.0)
        bc_r = np.random.normal(200.0, 12.0)
        if bc_r <= bc_l + 5.0:
            bc_r = bc_l + np.random.uniform(40.0, 120.0)

        amplitude = np.random.uniform(0.9, 1.4)
        mode_n = np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
        T0 = bc_l + (bc_r - bc_l) * x_norm
        T0 += amplitude * abs(bc_r - bc_l + 1e-6) * np.sin(mode_n * np.pi * x_norm)
        T0[0] = bc_l
        T0[-1] = bc_r
        return T0, bc_l, bc_r
        
    def compute_spatial_laplacian_star(self, T_star: np.ndarray) -> np.ndarray:
        """
        Compute dimensionless Laplacian ∇*²T* using finite differences.
        
        Args:
            T_star: Dimensionless temperature [batch, grid] or [grid]
            
        Returns:
            d2T_star_dx_star2: [batch, grid]
        """
        if T_star.ndim == 1:
            T_star = T_star[np.newaxis, :]
        
        d2T = np.zeros_like(T_star)
        d2T[:, 1:-1] = (T_star[:, 2:] - 2 * T_star[:, 1:-1] + T_star[:, :-2]) / (self.dx_star ** 2)
        d2T[:, 0] = d2T[:, 1]
        d2T[:, -1] = d2T[:, -2]
        return d2T
    
    def compute_heat_source_star(self, T_star: np.ndarray, T_prev_star: np.ndarray,
                                  T_prev2_star: np.ndarray, C: np.ndarray,
                                  alpha_star: np.ndarray) -> np.ndarray:
        """
        Compute dimensionless heat source Q̂ consistent with the Cattaneo equation.

        Correct nondimensional residual (dimensional PDE ÷ ΔT/dt):
            (τ/dt)·d2T* + dT* = Fo · sec_diff(T*) + Q̂

        Solving for Q̂:
            Q̂ = (τ/dt)·d2T* + dT* − Fo · sec_diff(T*)

        where sec_diff = T*_{j+1} − 2T*_j + T*_{j-1} (NOT divided by dx*²).

        Args:
            T_star, T_prev_star, T_prev2_star: Dimensionless temperatures [batch, grid]
            C: Cattaneo number [batch, grid] (τ = C · t_c)
            alpha_star: Dimensionless diffusivity [batch, grid] (unused, kept for compat)

        Returns:
            Q_star: Dimensionless heat source [batch, grid]
        """
        d2T = T_star - 2 * T_prev_star + T_prev2_star
        dT = T_star - T_prev_star

        # Spatial second difference (NOT divided by dx*²)
        if T_star.ndim == 1:
            T_star_2d = T_star[np.newaxis, :]
        else:
            T_star_2d = T_star
        sec_diff = np.zeros_like(T_star_2d)
        sec_diff[:, 1:-1] = T_star_2d[:, 2:] - 2 * T_star_2d[:, 1:-1] + T_star_2d[:, :-2]
        sec_diff[:, 0] = sec_diff[:, 1]
        sec_diff[:, -1] = sec_diff[:, -2]
        if T_star.ndim == 1:
            sec_diff = sec_diff[0]

        # τ/dt where τ = C · t_c and dt is the batch timestep
        # Note: self.Fo = α·dt/dx², and the data stores dt_batch
        tau_over_dt = C * self.t_c / (self.dt + 1e-30)

        Q_star = tau_over_dt * d2T + dT - self.Fo * sec_diff

        return Q_star
      
    
    def generate_temporal_sequence(self, n_samples: int, seq_len: int,
                                   timestep_jump: int = 1) -> Dict[str, np.ndarray]:
        """
        Generate temporal sequences entirely in dimensionless space.
        
        Creates smooth T*(x*, t*) fields where adjacent timesteps differ by 
        O(Fo) ~ O(1e-4), which is physically consistent.
        
        Args:
            n_samples: Number of sequences
            seq_len: Length of each sequence
            timestep_jump: Factor for multi-step prediction
            
        Returns:
            dict with dimensionless temperature sequences and parameters
        """
        # Sample random tau; alpha and rho_cp come from the material file
        C = np.random.uniform(self.C_range[0], self.C_range[1],
                             (n_samples, self.grid_size))
        alpha_star = np.full((n_samples, self.grid_size),
                             self.alpha_mat / self.alpha_ref)
        rho_cp = np.full((n_samples, self.grid_size), self.Cv_mat)
        
        # Sample base spatial profiles from GP: T* ~ O(1)
        base_profiles = self.gp_sampler.sample(n_samples, temp_range=(-2.0, 3.0))
        
        # Generate temporally-smooth perturbations
        # For dT*/dt* to be O(1), each timestep should change by O(dt_star)
        # Total variation over sequence ~ seq_len * dt_star
        n_keyframes = max(3, seq_len // 2)
        perturbation_keyframes = np.random.randn(n_samples, n_keyframes, self.grid_size)
        
        # Smooth perturbations spatially using GP covariance
        # Batched matrix-vector multiply: L @ perturbation_keyframes[i, k, :]
        # for all i, k simultaneously via einsum
        perturbation_keyframes = np.einsum('gj,nkj->nkg', self.gp_sampler.L, perturbation_keyframes)
        
        # Normalize perturbations so their std is 1, then scale to physical level
        for i in range(n_samples):
            std = perturbation_keyframes[i].std()
            if std > 0:
                perturbation_keyframes[i] /= std
        
        # Scale: total variation over sequence should be O(seq_len * Fo)
        # so that per-step changes are O(Fo) and dT*/d(step) ~ O(1)
        temporal_scale = seq_len * self.Fo
        perturbation_keyframes *= temporal_scale
        
        # Interpolate to get smooth temporal evolution
        # CubicSpline supports multi-column y arrays: operates on all grid points at once
        T_star_seq = np.zeros((n_samples, seq_len, self.grid_size))
        keyframe_indices = np.linspace(0, seq_len - 1, n_keyframes)
        eval_indices = np.arange(seq_len)

        for i in range(n_samples):
            interp_func = CubicSpline(
                keyframe_indices,
                perturbation_keyframes[i],  # [n_keyframes, grid_size]
                bc_type='natural'
            )
            T_star_seq[i] = base_profiles[i] + interp_func(eval_indices)  # [seq_len, grid_size]
        
        return {
            'T_star_seq': T_star_seq,
            'C': C,
            'alpha_star': alpha_star,
            'rho_cp': rho_cp
        }
    
    def generate_training_batch(self, n_samples: int, 
                                timestep_jump: int = 10) -> Dict[str, torch.Tensor]:
        """
        Generate a complete training batch via FDM-forward simulation.
        
        Instead of reverse-engineering Q from T_target (which makes the
        physics loss trivially zero), we:
          1. Sample smooth GP initial conditions
          2. Run the FDM solver forward for timestep_jump steps
          3. Record (T_n, T_nm1) → T_target as genuine PDE evolution
        
        Material consistency:
            alpha and rho_cp are derived from the material file (not random)
            so they match what the FDM solver actually uses.  Only tau is
            randomly varied.
        
        Args:
            n_samples: Number of samples
            timestep_jump: Number of timesteps to predict ahead
            
        Returns:
            dict with all inputs and targets as dimensional tensors
        """
        from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

        # Random tau (the only material parameter the solver accepts)
        C = np.random.uniform(self.C_range[0], self.C_range[1],
                              (n_samples, self.grid_size))
        tau_dim = C * self.t_c                                      # [n_samples, grid]

        # alpha and rho_cp derived from material file (match the solver)
        alpha_dim = np.full((n_samples, self.grid_size), self.alpha_mat)
        rho_cp = np.full((n_samples, self.grid_size), self.Cv_mat)

        # Sample GP-based initial conditions (dimensional)
        T_star_samples = self.gp_sampler.sample(n_samples, temp_range=(-2.0, 3.0))
        T0_all = self.T_ref + T_star_samples * self.delta_T

        # Random BCs
        bc_left_star = np.random.uniform(-2.0, 3.0, n_samples)
        bc_right_star = np.random.uniform(-2.0, 3.0, n_samples)
        bc_left = self.T_ref + bc_left_star * self.delta_T
        bc_right = self.T_ref + bc_right_star * self.delta_T

        T_n_all = np.zeros((n_samples, self.grid_size))
        T_nm1_all = np.zeros((n_samples, self.grid_size))
        T_target_all = np.zeros((n_samples, self.grid_size))
        q_all = np.zeros((n_samples, self.grid_size))

        # Prepare per-sample ICs with BCs applied
        for i in range(n_samples):
            T0_all[i, 0] = bc_left[i]
            T0_all[i, -1] = bc_right[i]

        # Parallel FDM evolution: 3×timestep_jump steps to get
        # (T_nm1, T_n, T_target) with physical velocity history
        args_list = [
            (T0_all[i], tau_dim[i], self.grid_size, self.dx, self.dt,
             timestep_jump, bc_left[i], bc_right[i])
            for i in range(n_samples)
        ]
        with mp.Pool(_NUM_WORKERS) as pool:
            results = pool.starmap(_fdm_evolve_pair, args_list)

        T_nm1_all = np.array([r[0] for r in results])
        T_n_all   = np.array([r[1] for r in results])
        T_target_all = np.array([r[2] for r in results])

        batch = {
            'T_n': torch.from_numpy(T_n_all).double(),
            'T_nm1': torch.from_numpy(T_nm1_all).double(),
            'T_target': torch.from_numpy(T_target_all).double(),
            'T_prev': torch.from_numpy(T_n_all).double(),
            'T_prev2': torch.from_numpy(T_nm1_all).double(),
            'q': torch.from_numpy(q_all).double(),
            'tau': torch.from_numpy(tau_dim).double(),
            'alpha': torch.from_numpy(alpha_dim).double(),
            'rho_cp': torch.from_numpy(rho_cp).double(),
            'bc_left': torch.from_numpy(bc_left).double(),
            'bc_right': torch.from_numpy(bc_right).double(),
            'dt': self.dt * timestep_jump,
            'dx': self.dx
        }
        
        return batch
    
    def generate_validation_set(self, n_samples: int, 
                                timestep_jump: int = 10) -> Dict[str, torch.Tensor]:
        """Generate validation set with different random seed."""
        np.random.seed(42)  # Fixed seed for reproducibility
        return self.generate_training_batch(n_samples, timestep_jump)

    def generate_step_bc_batch(self, n_samples: int,
                                timestep_jump: int = 10) -> Dict[str, torch.Tensor]:
        """
        Generate samples with step-function BCs: uniform or smooth interior
        with BCs that differ significantly.  This teaches the model that
        boundary influence propagates slowly and should NOT spread across
        the whole domain in one step.

        The target is computed via FDM, not reverse-engineering.
        """
        from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

        # Random tau (only parameter the solver accepts)
        C = np.random.uniform(self.C_range[0], self.C_range[1],
                              (n_samples, self.grid_size))
        tau_dim = C * self.t_c

        # alpha and rho_cp from material file (match solver)
        alpha_dim = np.full((n_samples, self.grid_size), self.alpha_mat)
        rho_cp = np.full((n_samples, self.grid_size), self.Cv_mat)

        # Random interior temperatures and BCs
        T_interior_star = np.random.uniform(-1.5, 2.5, n_samples)
        bc_left_star = np.random.uniform(-2.0, 3.0, n_samples)
        bc_right_star = np.random.uniform(-2.0, 3.0, n_samples)

        # Dimensional
        T_int = self.T_ref + T_interior_star * self.delta_T
        bc_left = self.T_ref + bc_left_star * self.delta_T
        bc_right = self.T_ref + bc_right_star * self.delta_T

        # Build initial conditions – mostly uniform with optional small GP noise
        T0_all = np.zeros((n_samples, self.grid_size))

        for i in range(n_samples):
            # Small GP perturbation on top of uniform interior (optional, 50% chance)
            if np.random.rand() < 0.5:
                noise = self.gp_sampler.sample(1, temp_range=(-0.05, 0.05))[0]
                T0_all[i] = T_int[i] + noise * self.delta_T
            else:
                T0_all[i] = T_int[i]

            # Apply BCs at boundaries (FDM solver expects this)
            T0_all[i, 0] = bc_left[i]
            T0_all[i, -1] = bc_right[i]

        # Parallel FDM evolution: 3×timestep_jump for velocity history
        args_list = [
            (T0_all[i], tau_dim[i], self.grid_size, self.dx, self.dt,
             timestep_jump, bc_left[i], bc_right[i])
            for i in range(n_samples)
        ]
        with mp.Pool(_NUM_WORKERS) as pool:
            results = pool.starmap(_fdm_evolve_pair, args_list)

        T_nm1_all = np.array([r[0] for r in results])
        T_n_all   = np.array([r[1] for r in results])
        T_target_all = np.array([r[2] for r in results])
        q_all = np.zeros((n_samples, self.grid_size))

        batch = {
            'T_n': torch.from_numpy(T_n_all).double(),
            'T_nm1': torch.from_numpy(T_nm1_all).double(),
            'T_target': torch.from_numpy(T_target_all).double(),
            'T_prev': torch.from_numpy(T_n_all).double(),
            'T_prev2': torch.from_numpy(T_nm1_all).double(),
            'q': torch.from_numpy(q_all).double(),
            'tau': torch.from_numpy(tau_dim).double(),
            'alpha': torch.from_numpy(alpha_dim).double(),
            'rho_cp': torch.from_numpy(rho_cp).double(),
            'bc_left': torch.from_numpy(bc_left).double(),
            'bc_right': torch.from_numpy(bc_right).double(),
            'dt': self.dt * timestep_jump,
            'dx': self.dx
        }
        return batch

    def generate_equilibrium_batch(self, n_samples: int,
                                    timestep_jump: int = 10) -> Dict[str, torch.Tensor]:
        """
        Generate equilibrium / near-equilibrium samples where the correct
        increment is exactly or nearly zero.

        Types of equilibrium profiles:
        1. Uniform temperature with matching BCs (T_target = T_n exactly)
        2. Linear profiles (steady-state of diffusion) with matching BCs
        3. Profiles where BCs match boundaries (smooth GP, BCs = field values)
        4. Demo-like scenarios (T=10K, bc=[10,50]) run through FDM

        These samples teach the model to output near-zero increments when
        the field is already at or near equilibrium, preventing systematic
        bias that causes autoregressive drift.
        """
        from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

        # Random tau (only parameter the solver accepts)
        C = np.random.uniform(self.C_range[0], self.C_range[1],
                              (n_samples, self.grid_size))
        tau_dim = C * self.t_c

        # alpha and rho_cp from material file (match solver)
        alpha_dim = np.full((n_samples, self.grid_size), self.alpha_mat)
        rho_cp = np.full((n_samples, self.grid_size), self.Cv_mat)

        T_n_all = np.zeros((n_samples, self.grid_size))
        T_nm1_all = np.zeros((n_samples, self.grid_size))
        T_target_all = np.zeros((n_samples, self.grid_size))
        q_all = np.zeros((n_samples, self.grid_size))
        bc_left_all = np.zeros(n_samples)
        bc_right_all = np.zeros(n_samples)

        # Build ICs for all samples; track which need FDM
        fdm_indices = []

        for i in range(n_samples):
            profile_type = np.random.choice(
                ['uniform', 'linear', 'smooth_gp', 'demo_like'],
                p=[0.35, 0.25, 0.15, 0.25]
            )

            if profile_type == 'uniform':
                # Uniform temperature: T = const, BCs match → zero increment
                T_val_star = np.random.uniform(-2.0, 3.0)
                T_val = self.T_ref + T_val_star * self.delta_T
                T0 = np.ones(self.grid_size) * T_val
                bc_left_all[i] = T_val
                bc_right_all[i] = T_val
                # True target = T_n (zero increment)
                T_target_all[i] = T0
                T_n_all[i] = T0
                T_nm1_all[i] = T0
                q_all[i] = 0.0
                continue

            elif profile_type == 'linear':
                T_left_star = np.random.uniform(-2.0, 3.0)
                T_right_star = np.random.uniform(-2.0, 3.0)
                T_left = self.T_ref + T_left_star * self.delta_T
                T_right = self.T_ref + T_right_star * self.delta_T
                T0 = np.linspace(T_left, T_right, self.grid_size)
                bc_left_all[i] = T_left
                bc_right_all[i] = T_right

            elif profile_type == 'smooth_gp':
                T0_star = self.gp_sampler.sample(1, temp_range=(-2.0, 3.0))[0]
                T0 = self.T_ref + T0_star * self.delta_T
                bc_left_all[i] = T0[0]
                bc_right_all[i] = T0[-1]

            else:  # demo_like
                T_int_star = np.random.uniform(-2.0, 3.0)
                T_int = self.T_ref + T_int_star * self.delta_T
                bc_l_star = np.random.uniform(-2.0, 3.0)
                bc_r_star = np.random.uniform(-2.0, 3.0)
                T0 = np.ones(self.grid_size) * T_int
                T0[0] = self.T_ref + bc_l_star * self.delta_T
                T0[-1] = self.T_ref + bc_r_star * self.delta_T
                bc_left_all[i] = T0[0]
                bc_right_all[i] = T0[-1]

            T_n_all[i] = T0
            T_nm1_all[i] = T0
            fdm_indices.append(i)

        # Parallel FDM for non-uniform samples: 3×timestep_jump for velocity history
        if fdm_indices:
            args_list = [
                (T_n_all[i], tau_dim[i], self.grid_size, self.dx, self.dt,
                 timestep_jump, bc_left_all[i], bc_right_all[i])
                for i in fdm_indices
            ]
            with mp.Pool(_NUM_WORKERS) as pool:
                results = pool.starmap(_fdm_evolve_pair, args_list)
            for j, i in enumerate(fdm_indices):
                T_nm1_all[i] = results[j][0]
                T_n_all[i]   = results[j][1]
                T_target_all[i] = results[j][2]

        batch = {
            'T_n': torch.from_numpy(T_n_all).double(),
            'T_nm1': torch.from_numpy(T_nm1_all).double(),
            'T_target': torch.from_numpy(T_target_all).double(),
            'T_prev': torch.from_numpy(T_n_all).double(),
            'T_prev2': torch.from_numpy(T_nm1_all).double(),
            'q': torch.from_numpy(q_all).double(),
            'tau': torch.from_numpy(tau_dim).double(),
            'alpha': torch.from_numpy(alpha_dim).double(),
            'rho_cp': torch.from_numpy(rho_cp).double(),
            'bc_left': torch.from_numpy(bc_left_all).double(),
            'bc_right': torch.from_numpy(bc_right_all).double(),
            'dt': self.dt * timestep_jump,
            'dx': self.dx
        }
        return batch

    def generate_sinusoidal_batch(self, n_samples: int,
                                   timestep_jump: int = 10) -> Dict[str, torch.Tensor]:
        """
        Generate samples with sinusoidal perturbation ICs on a linear baseline.

        IC: T(x) = bc_left + (bc_right - bc_left)*x/L
                    + A * (bc_right - bc_left) * sin(n*pi*x/L)

        This matches the benchmark scenario (linear + sine) and trains the
        model on the kinds of non-equilibrium profiles it will see during
        long autoregressive rollouts.  Amplitude A and mode number n are
        varied randomly.

        Also uses wider BC ranges and varied amplitudes to improve
        generalisation across different operating conditions.
        """
        from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

        # Random tau
        C = np.random.uniform(self.C_range[0], self.C_range[1],
                              (n_samples, self.grid_size))
        tau_dim = C * self.t_c

        alpha_dim = np.full((n_samples, self.grid_size), self.alpha_mat)
        rho_cp = np.full((n_samples, self.grid_size), self.Cv_mat)

        T_n_all = np.zeros((n_samples, self.grid_size))
        T_nm1_all = np.zeros((n_samples, self.grid_size))
        q_all = np.zeros((n_samples, self.grid_size))
        bc_left_all = np.zeros(n_samples)
        bc_right_all = np.zeros(n_samples)

        x_norm = np.linspace(0, 1, self.grid_size)

        for i in range(n_samples):
            if np.random.rand() < 0.7:
                T0, bc_l, bc_r = self._sample_benchmark_relaxation_ic()
            else:
                # Random BCs with wider range  (includes benchmark 100/200 scenario)
                bc_l_star = np.random.uniform(-2.0, 3.0)
                bc_r_star = np.random.uniform(-2.0, 3.0)
                bc_l = self.T_ref + bc_l_star * self.delta_T
                bc_r = self.T_ref + bc_r_star * self.delta_T

                # Linear baseline + sinusoidal perturbation
                amplitude = np.random.uniform(0.1, 1.5)   # 10–150% of ΔBC
                mode_n = np.random.choice([1, 2, 3], p=[0.6, 0.25, 0.15])
                T0 = bc_l + (bc_r - bc_l) * x_norm
                T0 += amplitude * abs(bc_r - bc_l + 1e-6) * np.sin(mode_n * np.pi * x_norm)

            # Enforce BCs exactly
            T0[0] = bc_l
            T0[-1] = bc_r

            T_n_all[i] = T0
            T_nm1_all[i] = T0
            bc_left_all[i] = bc_l
            bc_right_all[i] = bc_r

        # Parallel FDM evolution: 3×timestep_jump for velocity history
        args_list = [
            (T_n_all[i], tau_dim[i], self.grid_size, self.dx, self.dt,
             timestep_jump, bc_left_all[i], bc_right_all[i])
            for i in range(n_samples)
        ]
        with mp.Pool(_NUM_WORKERS) as pool:
            results = pool.starmap(_fdm_evolve_pair, args_list)

        T_nm1_all = np.array([r[0] for r in results])
        T_n_all   = np.array([r[1] for r in results])
        T_target_all = np.array([r[2] for r in results])

        batch = {
            'T_n': torch.from_numpy(T_n_all).double(),
            'T_nm1': torch.from_numpy(T_nm1_all).double(),
            'T_target': torch.from_numpy(T_target_all).double(),
            'T_prev': torch.from_numpy(T_n_all).double(),
            'T_prev2': torch.from_numpy(T_nm1_all).double(),
            'q': torch.from_numpy(q_all).double(),
            'tau': torch.from_numpy(tau_dim).double(),
            'alpha': torch.from_numpy(alpha_dim).double(),
            'rho_cp': torch.from_numpy(rho_cp).double(),
            'bc_left': torch.from_numpy(bc_left_all).double(),
            'bc_right': torch.from_numpy(bc_right_all).double(),
            'dt': self.dt * timestep_jump,
            'dx': self.dx
        }
        return batch


    def generate_trajectory_batch(self, n_trajectories: int,
                                    n_pairs_per_traj: int = 10,
                                    total_jumps: int = 100,
                                    timestep_jump: int = 10) -> Dict[str, torch.Tensor]:
        """
        Generate multi-step FDM trajectories and extract training pairs from
        intermediate states along each trajectory.

        This is critical for autoregressive stability: the model must see
        intermediate states (not just pristine ICs) during training, because
        during rollout the model's own predictions become the next input.

        Args:
            n_trajectories: Number of FDM trajectories to simulate
            n_pairs_per_traj: Training pairs to extract per trajectory
            total_jumps: Total number of neural-step-sized jumps per trajectory
            timestep_jump: FDM sub-steps per neural step

        Returns:
            Batch dict with n_trajectories * n_pairs_per_traj samples
        """
        from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

        n_samples = n_trajectories * n_pairs_per_traj

        # Pre-generate per-trajectory parameters
        traj_params = []
        for traj in range(n_trajectories):
            C = np.random.uniform(self.C_range[0], self.C_range[1], self.grid_size)
            tau_dim = C * self.t_c
            alpha_dim = np.full(self.grid_size, self.alpha_mat)
            rho_cp = np.full(self.grid_size, self.Cv_mat)

            bc_l_star = np.random.uniform(-2.0, 3.0)
            bc_r_star = np.random.uniform(-2.0, 3.0)
            bc_l = self.T_ref + bc_l_star * self.delta_T
            bc_r = self.T_ref + bc_r_star * self.delta_T

            x_norm = np.linspace(0, 1, self.grid_size)
            if np.random.rand() < 0.92:
                T0, bc_l, bc_r = self._sample_benchmark_relaxation_ic()
            elif np.random.rand() < 0.98:
                amplitude = np.random.uniform(0.1, 1.5)
                mode_n = np.random.choice([1, 2, 3], p=[0.6, 0.25, 0.15])
                T0 = bc_l + (bc_r - bc_l) * x_norm
                T0 += amplitude * abs(bc_r - bc_l + 1e-6) * np.sin(mode_n * np.pi * x_norm)
            else:
                T_int_star = np.random.uniform(-1.5, 2.5)
                T_int = self.T_ref + T_int_star * self.delta_T
                T0 = np.ones(self.grid_size) * T_int

            T0[0] = bc_l
            T0[-1] = bc_r

            traj_params.append((T0, tau_dim, alpha_dim, rho_cp, bc_l, bc_r))

        if self._use_gpu_trajectory_backend(n_trajectories, total_jumps):
            T0_batch = np.stack([p[0] for p in traj_params], axis=0)
            tau_batch = np.stack([p[1] for p in traj_params], axis=0)
            bc_left_batch = np.array([p[4] for p in traj_params], dtype=np.float64)
            bc_right_batch = np.array([p[5] for p in traj_params], dtype=np.float64)
            all_trajectories = _fdm_trajectory_batch_gpu(
                T0_batch, tau_batch, self.dx, self.dt, timestep_jump,
                bc_left_batch, bc_right_batch, total_jumps,
                self.Cv_mat, self.k_mat,
            )
        else:
            args_list = [
                (p[0], p[1], self.grid_size, self.dx, self.dt, timestep_jump,
                 p[4], p[5], total_jumps)
                for p in traj_params
            ]
            with mp.Pool(_NUM_WORKERS) as pool:
                all_trajectories = pool.starmap(_fdm_trajectory, args_list)

        # Extract training pairs from trajectories
        T_n_all = np.zeros((n_samples, self.grid_size))
        T_nm1_all = np.zeros((n_samples, self.grid_size))
        T_target_all = np.zeros((n_samples, self.grid_size))
        q_all = np.zeros((n_samples, self.grid_size))
        tau_all = np.zeros((n_samples, self.grid_size))
        alpha_all = np.zeros((n_samples, self.grid_size))
        rho_cp_all = np.zeros((n_samples, self.grid_size))
        bc_left_all = np.zeros(n_samples)
        bc_right_all = np.zeros(n_samples)

        sample_idx = 0
        for traj_idx, trajectory in enumerate(all_trajectories):
            T0, tau_dim, alpha_dim, rho_cp, bc_l, bc_r = traj_params[traj_idx]

            max_start = total_jumps - 1
            if max_start < 1:
                max_start = 1
            # Skip the earliest rollout region where one-step changes are
            # often too small and encourage a near-identity mapping.
            min_start = min(20, max_start)
            indices = np.sort(np.random.choice(
                range(min_start, max_start + 1), size=n_pairs_per_traj, replace=True
            ))

            for t in indices:
                T_n_all[sample_idx] = trajectory[t]
                T_nm1_all[sample_idx] = trajectory[t - 1]
                T_target_all[sample_idx] = trajectory[t + 1]
                q_all[sample_idx] = 0.0
                tau_all[sample_idx] = tau_dim
                alpha_all[sample_idx] = alpha_dim
                rho_cp_all[sample_idx] = rho_cp
                bc_left_all[sample_idx] = bc_l
                bc_right_all[sample_idx] = bc_r
                sample_idx += 1

        print(f"    Generated {n_trajectories} trajectories ({sample_idx} pairs)")

        batch = {
            'T_n': torch.from_numpy(T_n_all[:sample_idx]).double(),
            'T_nm1': torch.from_numpy(T_nm1_all[:sample_idx]).double(),
            'T_target': torch.from_numpy(T_target_all[:sample_idx]).double(),
            'T_prev': torch.from_numpy(T_n_all[:sample_idx]).double(),
            'T_prev2': torch.from_numpy(T_nm1_all[:sample_idx]).double(),
            'q': torch.from_numpy(q_all[:sample_idx]).double(),
            'tau': torch.from_numpy(tau_all[:sample_idx]).double(),
            'alpha': torch.from_numpy(alpha_all[:sample_idx]).double(),
            'rho_cp': torch.from_numpy(rho_cp_all[:sample_idx]).double(),
            'bc_left': torch.from_numpy(bc_left_all[:sample_idx]).double(),
            'bc_right': torch.from_numpy(bc_right_all[:sample_idx]).double(),
            'dt': self.dt * timestep_jump,
            'dx': self.dx
        }
        return batch

    def generate_trajectory_sequences(self, n_trajectories: int,
                                       total_jumps: int = 200,
                                       timestep_jump: int = 10) -> Dict[str, torch.Tensor]:
        """
        Generate full FDM trajectory arrays for ground-truth rollout training.

        Unlike generate_trajectory_batch() which extracts random (T_n, T_nm1, T_target)
        pairs, this method returns the FULL trajectory as a 3D tensor so the trainer
        can unroll the model autoregressively and compare against ground truth at
        every step.

        Args:
            n_trajectories: Number of FDM trajectories to simulate
            total_jumps: Total neural-step-sized jumps per trajectory
            timestep_jump: FDM sub-steps per neural step

        Returns:
            dict with:
                'trajectories': [N_traj, total_jumps+1, grid_size] float tensor
                'tau', 'alpha', 'rho_cp': [N_traj, grid_size]
                'q': [N_traj, grid_size] (zeros)
                'bc_left', 'bc_right': [N_traj]
                'dt': float, 'dx': float
        """
        from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

        all_trajs = np.zeros((n_trajectories, total_jumps + 1, self.grid_size))
        tau_all = np.zeros((n_trajectories, self.grid_size))
        alpha_all = np.zeros((n_trajectories, self.grid_size))
        rho_cp_all = np.zeros((n_trajectories, self.grid_size))
        q_all = np.zeros((n_trajectories, self.grid_size))
        bc_left_all = np.zeros(n_trajectories)
        bc_right_all = np.zeros(n_trajectories)

        grid = np.ones(self.grid_size, dtype=int)

        # Pre-generate per-trajectory parameters
        traj_params = []
        for traj in range(n_trajectories):
            C = np.random.uniform(self.C_range[0], self.C_range[1], self.grid_size)
            tau_dim = C * self.t_c
            alpha_dim = np.full(self.grid_size, self.alpha_mat)
            rho_cp = np.full(self.grid_size, self.Cv_mat)

            bc_l_star = np.random.uniform(-2.0, 3.0)
            bc_r_star = np.random.uniform(-2.0, 3.0)
            bc_l = self.T_ref + bc_l_star * self.delta_T
            bc_r = self.T_ref + bc_r_star * self.delta_T

            x_norm = np.linspace(0, 1, self.grid_size)
            if np.random.rand() < 0.94:
                T0, bc_l, bc_r = self._sample_benchmark_relaxation_ic()
            elif np.random.rand() < 0.99:
                amplitude = np.random.uniform(0.1, 1.5)
                mode_n = np.random.choice([1, 2, 3], p=[0.6, 0.25, 0.15])
                T0 = bc_l + (bc_r - bc_l) * x_norm
                T0 += amplitude * abs(bc_r - bc_l + 1e-6) * np.sin(mode_n * np.pi * x_norm)
            else:
                T_int_star = np.random.uniform(-1.5, 2.5)
                T_int = self.T_ref + T_int_star * self.delta_T
                T0 = np.ones(self.grid_size) * T_int

            T0[0] = bc_l
            T0[-1] = bc_r
            traj_params.append((T0, tau_dim, alpha_dim, rho_cp, bc_l, bc_r))

        if self._use_gpu_trajectory_backend(n_trajectories, total_jumps):
            T0_batch = np.stack([p[0] for p in traj_params], axis=0)
            tau_batch = np.stack([p[1] for p in traj_params], axis=0)
            bc_left_batch = np.array([p[4] for p in traj_params], dtype=np.float64)
            bc_right_batch = np.array([p[5] for p in traj_params], dtype=np.float64)
            all_trajectory_lists = _fdm_trajectory_batch_gpu(
                T0_batch, tau_batch, self.dx, self.dt, timestep_jump,
                bc_left_batch, bc_right_batch, total_jumps,
                self.Cv_mat, self.k_mat,
            )
        else:
            args_list = [
                (p[0], p[1], self.grid_size, self.dx, self.dt, timestep_jump,
                 p[4], p[5], total_jumps)
                for p in traj_params
            ]
            with mp.Pool(_NUM_WORKERS) as pool:
                all_trajectory_lists = pool.starmap(_fdm_trajectory, args_list)

        # Pack into arrays
        for traj_idx, trajectory in enumerate(all_trajectory_lists):
            T0, tau_dim, alpha_dim, rho_cp, bc_l, bc_r = traj_params[traj_idx]
            for j, state in enumerate(trajectory):
                all_trajs[traj_idx, j] = state
            tau_all[traj_idx] = tau_dim
            alpha_all[traj_idx] = alpha_dim
            rho_cp_all[traj_idx] = rho_cp
            bc_left_all[traj_idx] = bc_l
            bc_right_all[traj_idx] = bc_r

        print(f"    Generated {n_trajectories} trajectory sequences")

        return {
            'trajectories': torch.from_numpy(all_trajs).double(),
            'tau': torch.from_numpy(tau_all).double(),
            'alpha': torch.from_numpy(alpha_all).double(),
            'rho_cp': torch.from_numpy(rho_cp_all).double(),
            'q': torch.from_numpy(q_all).double(),
            'bc_left': torch.from_numpy(bc_left_all).double(),
            'bc_right': torch.from_numpy(bc_right_all).double(),
            'dt': self.dt * timestep_jump,
            'dx': self.dx,
        }


def generate_training_data(grid_size: int = 112, dx: float = 1e-8, dt: float = 1e-13,
                          n_train: int = 1000, n_val: int = 200,
                          timestep_jump: int = 1,
                          trajectory_total_jumps: int = 200,
                          fdm_backend: str = 'auto',
                          save_path: Optional[str] = None) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Generate complete training and validation datasets.

    Args:
        grid_size: Number of grid cells
        dx: Spatial step
        dt: Time step
        n_train: Number of training samples
        n_val: Number of validation samples
        timestep_jump: FDM steps per supervised pair (default 1 for single-step prediction)
        trajectory_total_jumps: Total FDM steps per trajectory sequence
        save_path: Optional path to save datasets

    Returns:
        train_data, val_data, train_trajectories, val_trajectories
    """
    print("Generating training data...")
    generator = CattaneoDataGenerator(grid_size, dx, dt, fdm_backend=fdm_backend)
    
    # Bias strongly toward genuinely evolving benchmark-style transients.
    # Keep only a token near-equilibrium component so the model does not
    # collapse into a near-identity integrator dominated by tiny updates.
    n_gp = int(n_train * 0.03)
    n_step = int(n_train * 0.07)
    n_eq = max(1, int(n_train * 0.005))
    n_sin = int(n_train * 0.45)
    n_traj = n_train - n_gp - n_step - n_eq - n_sin
    
    # Generate GP-based samples in batches
    batch_size = 100
    train_batches = []
    
    for i in range(0, n_gp, batch_size):
        current_batch_size = min(batch_size, n_gp - i)
        batch = generator.generate_training_batch(current_batch_size, timestep_jump)
        train_batches.append(batch)
        print(f"  Generated {min(i + batch_size, n_gp)}/{n_gp} GP training samples")
    
    # Generate step-BC samples (FDM-based, slower but critical for generalization)
    if n_step > 0:
        print(f"  Generating {n_step} step-BC training samples (FDM)...")
        for i in range(0, n_step, batch_size):
            current_batch_size = min(batch_size, n_step - i)
            batch = generator.generate_step_bc_batch(current_batch_size, timestep_jump)
            train_batches.append(batch)
            print(f"  Generated {min(i + batch_size, n_step)}/{n_step} step-BC training samples")

    # Generate equilibrium samples (zero-increment, critical for autoregressive stability)
    if n_eq > 0:
        print(f"  Generating {n_eq} equilibrium training samples...")
        for i in range(0, n_eq, batch_size):
            current_batch_size = min(batch_size, n_eq - i)
            batch = generator.generate_equilibrium_batch(current_batch_size, timestep_jump)
            train_batches.append(batch)
            print(f"  Generated {min(i + batch_size, n_eq)}/{n_eq} equilibrium training samples")

    # Generate sinusoidal perturbation samples (matches benchmark IC scenario)
    if n_sin > 0:
        print(f"  Generating {n_sin} sinusoidal training samples (FDM)...")
        for i in range(0, n_sin, batch_size):
            current_batch_size = min(batch_size, n_sin - i)
            batch = generator.generate_sinusoidal_batch(current_batch_size, timestep_jump)
            train_batches.append(batch)
            print(f"  Generated {min(i + batch_size, n_sin)}/{n_sin} sinusoidal training samples")

    # Generate trajectory samples (multi-step FDM rollouts, critical for
    # autoregressive stability — model sees intermediate evolved states)
    if n_traj > 0:
        # Each trajectory produces n_pairs_per_traj samples
        n_pairs = 10
        n_trajectories = max(1, n_traj // n_pairs)
        actual_n_traj = n_trajectories * n_pairs
        print(f"  Generating {actual_n_traj} trajectory training samples "
              f"({n_trajectories} trajectories × {n_pairs} pairs)...")
        traj_batch = generator.generate_trajectory_batch(
            n_trajectories=n_trajectories,
            n_pairs_per_traj=n_pairs,
            total_jumps=trajectory_total_jumps,
            timestep_jump=timestep_jump
        )
        train_batches.append(traj_batch)
        print(f"  Generated {len(traj_batch['T_n'])} trajectory training samples")
    
    # Concatenate batches
    train_data = {
        key: torch.cat([batch[key] for batch in train_batches], dim=0)
        if isinstance(train_batches[0][key], torch.Tensor)
        else train_batches[0][key]
        for key in train_batches[0].keys()
    }
    
    # Validation should stress dynamic fidelity even more aggressively than train.
    print("Generating validation data...")
    np.random.seed(42)
    n_val_gp = int(n_val * 0.02)
    n_val_step = int(n_val * 0.08)
    n_val_eq = max(1, int(n_val * 0.01))
    n_val_sin = int(n_val * 0.50)
    n_val_traj = n_val - n_val_gp - n_val_step - n_val_eq - n_val_sin
    val_gp = generator.generate_training_batch(n_val_gp, timestep_jump)
    val_batches = [val_gp]
    if n_val_step > 0:
        val_step = generator.generate_step_bc_batch(n_val_step, timestep_jump)
        val_batches.append(val_step)
    if n_val_eq > 0:
        val_eq = generator.generate_equilibrium_batch(n_val_eq, timestep_jump)
        val_batches.append(val_eq)
    if n_val_sin > 0:
        val_sin = generator.generate_sinusoidal_batch(n_val_sin, timestep_jump)
        val_batches.append(val_sin)
    if n_val_traj > 0:
        n_val_pairs = 10
        n_val_trajectories = max(1, n_val_traj // n_val_pairs)
        val_traj = generator.generate_trajectory_batch(
            n_trajectories=n_val_trajectories,
            n_pairs_per_traj=n_val_pairs,
            total_jumps=trajectory_total_jumps,
            timestep_jump=timestep_jump
        )
        val_batches.append(val_traj)
    
    if len(val_batches) > 1:
        val_data = {
            key: torch.cat([b[key] for b in val_batches], dim=0)
            if isinstance(val_batches[0][key], torch.Tensor)
            else val_batches[0][key]
            for key in val_batches[0].keys()
        }
    else:
        val_data = val_gp

    # Generate full trajectory sequences for GT rollout training
    print("Generating trajectory sequences for rollout training...")
    n_train_traj_seq = max(1, n_train // 30)  # ~30 trajectories for 1000 samples
    n_val_traj_seq = max(1, n_val // 30)      # ~7 trajectories for 200 samples

    train_trajectories = generator.generate_trajectory_sequences(
        n_trajectories=n_train_traj_seq,
        total_jumps=trajectory_total_jumps,
        timestep_jump=timestep_jump
    )
    val_trajectories = generator.generate_trajectory_sequences(
        n_trajectories=n_val_traj_seq,
        total_jumps=trajectory_total_jumps,
        timestep_jump=timestep_jump
    )

    # Save if path provided
    if save_path is not None:
        print(f"Saving datasets to {save_path}...")
        torch.save({
            'train': train_data,
            'val': val_data,
            'train_trajectories': train_trajectories,
            'val_trajectories': val_trajectories,
        }, save_path)

    print("Data generation complete!")
    print(f"Training samples: {len(train_data['T_n'])}")
    print(f"Validation samples: {len(val_data['T_n'])}")
    print(f"Train trajectory sequences: {train_trajectories['trajectories'].shape}")
    print(f"Val trajectory sequences: {val_trajectories['trajectories'].shape}")

    return train_data, val_data, train_trajectories, val_trajectories


if __name__ == '__main__':
    # Test data generation
    grid_size = 112
    dx = 1e-8
    dt = 1e-13
    
    generator = CattaneoDataGenerator(grid_size, dx, dt)
    batch = generator.generate_training_batch(n_samples=4, timestep_jump=10)
    
    print("Generated batch shapes:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}, range: [{val.min():.2e}, {val.max():.2e}]")
        else:
            print(f"  {key}: {val}")
