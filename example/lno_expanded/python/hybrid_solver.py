"""
Hybrid Cattaneo Solver: Neural Surrogate + FDM Fallback
=======================================================

Integrates the Cattaneo-LNO neural surrogate with the cell-centered FDM solver.
Uses adaptive switching between neural and FDM based on:
- Stability conditions (CFL-like)
- Residual error estimates
- Gradient sharpness (shock detection)

Parallel execution: Neural on GPU, FDM on CPU (synchronized)
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

# Import existing FDM solver
from HF_Cattaneo import nl_solve_HF_1d_Cattaneo

# Import neural model
from cattaneo_lno import CattaneoLNO


@dataclass
class HybridSolverConfig:
    """Configuration for hybrid solver."""
    # Neural model parameters
    model_path: Optional[str] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Switching thresholds (dimensionless)
    residual_threshold: float = 0.05   # Training-aligned normalized residual RMS
    gradient_threshold: float = 500.0  # Dimensionless |dT*/dx*|, very high — let neural handle normal BCs
    cfl_safety_factor: float = 10.0  # High: neural trained on implicit solver data, CFL not a strict limit
    
    # Multi-step parameters
    timestep_jump: int = 1
    max_timestep_jump: int = 1        # Never exceed training timestep_jump
    
    # Adaptive parameters
    enable_adaptive_jump: bool = True
    error_history_size: int = 5
    
    # Fallback parameters
    fallback_on_failure: bool = True
    fdm_backend: str = 'auto'
    verbose: bool = False


class HybridCattaneoSolver:
    """
    Hybrid solver combining neural surrogate with FDM fallback.
    
    Strategy:
    1. Check if neural prediction is stable (CFL condition)
    2. Predict with neural network
    3. Estimate residual error
    4. If error > threshold or unstable, fall back to FDM
    5. Adapt timestep jump based on error history
    """
    
    def __init__(self, grid_size: int, dx: float, dt: float,
                 model: Optional[CattaneoLNO] = None,
                 config: Optional[HybridSolverConfig] = None):
        self.grid_size = grid_size
        self.dx = dx
        self.dt = dt
        self.config = config or HybridSolverConfig()
        
        # Neural model
        self.model = model
        if self.model is not None:
            self.model.to(self.config.device)
            self.model.eval()
        
        # State history for adaptive control
        self.error_history = deque(maxlen=self.config.error_history_size)
        self.current_jump = self.config.timestep_jump
        
        # Statistics
        self.stats = {
            'neural_calls': 0,
            'fdm_calls': 0,
            'fallbacks': 0,
            'total_steps': 0
        }

    @staticmethod
    def _steady_profile(grid_size: int, bc_left: float, bc_right: float) -> np.ndarray:
        xi = np.linspace(0.0, 1.0, grid_size)
        return bc_left + (bc_right - bc_left) * xi

    def violates_relaxation(self, T_new: np.ndarray, T: np.ndarray,
                            bc_left: float, bc_right: float,
                            tol: float = 1e-8) -> bool:
        """Disabled: Cattaneo (hyperbolic) dynamics can legitimately increase
        deviation from steady state during transient wave propagation.
        Always returns False so the model's prediction is used as-is."""
        return False

    def project_relaxing_step(self, T_new: np.ndarray, T: np.ndarray,
                              bc_left: float, bc_right: float) -> np.ndarray:
        steady = self._steady_profile(self.grid_size, bc_left, bc_right)
        delta = T_new - T
        relax_dir = steady - T
        step_norm = np.linalg.norm(delta)
        relax_norm = max(np.linalg.norm(relax_dir), 1e-12)
        gamma = min(step_norm / relax_norm, 1.0)
        T_proj = T + gamma * relax_dir
        T_proj[0] = bc_left
        T_proj[-1] = bc_right
        return T_proj
        
    def load_model(self, path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        state_dict = checkpoint['model_state_dict']
        use_spectral_norm = any(
            'pointwise.weight_orig' in k for k in state_dict.keys()
        )
        if self.model is None:
            from cattaneo_lno import create_cattaneo_model
            self.model = create_cattaneo_model(
                self.grid_size, use_spectral_norm=use_spectral_norm
            )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.config.device)
        self.model.eval()
        if self.config.verbose:
            print(f"Loaded model from {path}")
    
    def compute_gradient_sharpness(self, T: np.ndarray) -> float:
        """
        Detect sharp gradients (potential shocks) using dimensionless measure.
        
        Computes max |dT*/dx*| where T* and x* are dimensionless.
        For smooth fields this should be O(1), for shocks O(10+).
        
        Args:
            T: Temperature field (dimensional)
            
        Returns:
            dimensionless sharpness metric
        """
        # Use model's scaler if available for proper nondimensionalization
        if self.model is not None and hasattr(self.model, 'scaler'):
            scaler = self.model.scaler
            T_star = (T - scaler.T_ref) / scaler.delta_T
            dx_star = self.dx / scaler.L
        else:
            # Fallback: normalize by data range
            T_range = T.max() - T.min()
            T_star = (T - T.mean()) / (T_range + 1e-10)
            L = self.grid_size * self.dx
            dx_star = self.dx / L
        
        dTdx_star = np.abs(np.gradient(T_star, dx_star))
        return np.max(dTdx_star)
    
    def estimate_residual(self, T_new: np.ndarray, T: np.ndarray, 
                         T_prev: np.ndarray, q: np.ndarray,
                         tau: float, alpha: np.ndarray, 
                         rho_cp: np.ndarray) -> float:
        """
        Estimate the Cattaneo residual using the same normalized form as
        the training loss.
        
        Args:
            T_new: Predicted temperature (dimensional)
            T: Current temperature (dimensional)
            T_prev: Previous temperature (dimensional)
            q: Heat source (dimensional)
            tau: Relaxation time
            alpha: Thermal diffusivity
            rho_cp: ρ*c_p
            
        Returns:
            RMS residual in training-normalized units
        """
        if self.model is not None and hasattr(self.model, 'scaler'):
            scaler = self.model.scaler
            dt_eff = self.dt * self.current_jump
            T_new_star = (T_new - scaler.T_ref) / scaler.delta_T
            T_star = (T - scaler.T_ref) / scaler.delta_T
            T_prev_star = (T_prev - scaler.T_ref) / scaler.delta_T

            tau_field = np.asarray(tau)
            if tau_field.ndim == 0:
                tau_field = np.full_like(T_new_star, float(tau_field))

            Fo = np.clip(alpha * dt_eff / (self.dx ** 2), scaler.Fo_min, scaler.Fo_max)
            tau_over_dt = tau_field / (dt_eff + 1e-30)
            q_star = q / scaler.q_c

            sec_diff = np.empty_like(T_new_star)
            sec_diff[1:-1] = T_new_star[2:] - 2.0 * T_new_star[1:-1] + T_new_star[:-2]
            sec_diff[0] = T_new_star[1] - T_new_star[0]
            sec_diff[-1] = T_new_star[-2] - T_new_star[-1]

            d2T = T_new_star - 2.0 * T_star + T_prev_star
            dT = T_new_star - T_star
            raw_residual = tau_over_dt * d2T + dT - Fo * sec_diff - q_star
            residual = raw_residual / (1.0 + tau_over_dt)
            return float(np.sqrt(np.mean(residual ** 2)))
        else:
            # Fallback: use raw values (may not be well-scaled)
            T_new_star = T_new
            T_star = T
            T_prev_star = T_prev
            dt_star = self.dt * self.current_jump
            dx_star = self.dx
            C = tau
            alpha_star = alpha
            q_star = q
        
        # Weak-form residual (all terms small, O(dt_star))
        d2T = T_new_star - 2 * T_star + T_prev_star
        dT = T_new_star - T_star
        
        laplacian = np.zeros_like(T_new_star)
        laplacian[1:-1] = (T_new_star[2:] - 2 * T_new_star[1:-1] + T_new_star[:-2]) / (dx_star ** 2)
        laplacian[0] = laplacian[1]
        laplacian[-1] = laplacian[-2]
        
        residual = C * d2T + dt_star * dT - dt_star**2 * alpha_star * laplacian - q_star
        
        return np.sqrt(np.mean(residual ** 2))
    
    def check_stability(self, T: np.ndarray, tau: float, 
                       alpha: np.ndarray) -> bool:
        """
        Check CFL-like stability condition for Cattaneo equation.
        
        Uses dimensionless CFL: c* · dt* / dx* < threshold
        where c* = sqrt(α*/C) and C = τ/t_c.
        
        Args:
            T: Temperature field
            tau: Relaxation time
            alpha: Thermal diffusivity
            
        Returns:
            True if stable
        """
        if self.model is not None and hasattr(self.model, 'scaler'):
            scaler = self.model.scaler
            t_c = scaler.L ** 2 / scaler.alpha_ref
            alpha_star = alpha / scaler.alpha_ref
            C = tau / t_c
            c_star = np.sqrt(alpha_star / (C + 1e-15))
            dt_star = (self.dt * self.current_jump) / t_c
            dx_star = self.dx / scaler.L
            cfl = np.max(c_star) * dt_star / dx_star
        else:
            # Dimensional fallback
            c = np.sqrt(alpha / (tau + 1e-10))
            c_max = np.max(c)
            dt_effective = self.dt * self.current_jump
            cfl = c_max * dt_effective / self.dx
        
        stable = cfl < self.config.cfl_safety_factor
        
        if self.config.verbose and not stable:
            print(f"  Unstable: CFL = {cfl:.2f} > {self.config.cfl_safety_factor}")
        
        return stable
    
    def neural_predict(self, T: np.ndarray, T_prev: np.ndarray,
                      q: np.ndarray, tau: float, alpha: np.ndarray,
                      rho_cp: np.ndarray, bc_left: float, 
                      bc_right: float,
                      hidden_state=None) -> Tuple[np.ndarray, Dict]:
        """
        Predict next state using neural surrogate.
        
        Args:
            T: Current temperature
            T_prev: Previous temperature
            q: Heat source
            tau: Relaxation time
            alpha: Thermal diffusivity
            rho_cp: ρ*c_p
            bc_left, bc_right: Boundary conditions
            hidden_state: recurrent memory tensor or None
            
        Returns:
            T_pred, info dict (info['hidden_state'] carries the new memory)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to tensors — create tau directly on device
        T_t = torch.from_numpy(T).float().unsqueeze(0).to(self.config.device)
        T_prev_t = torch.from_numpy(T_prev).float().unsqueeze(0).to(self.config.device)
        q_t = torch.from_numpy(q).float().unsqueeze(0).to(self.config.device)
        tau_t = torch.full((1, self.grid_size), tau, device=self.config.device)
        alpha_t = torch.from_numpy(alpha).float().unsqueeze(0).to(self.config.device)
        rho_cp_t = torch.from_numpy(rho_cp).float().unsqueeze(0).to(self.config.device)
        bc_left_t = torch.tensor([bc_left], device=self.config.device)
        bc_right_t = torch.tensor([bc_right], device=self.config.device)
        
        # Predict — use effective dt (base dt * timestep_jump) matching training
        dt_effective = self.dt * self.current_jump
        with torch.inference_mode():
            output = self.model(
                T_t, T_prev_t, q_t, tau_t, alpha_t, rho_cp_t,
                bc_left_t, bc_right_t, dt_effective, self.dx,
                hidden_state=hidden_state,
            )
        
        # Extract prediction
        T_pred = output['T_pred'].squeeze(0).cpu().numpy()
        
        # Sanity clamp: keep predictions within a physically plausible range.
        # Training data T* ∈ (-2, 3) → T ∈ (T_ref - 2·ΔT, T_ref + 3·ΔT)
        if self.model is not None and hasattr(self.model, 'scaler'):
            s = self.model.scaler
            T_lo = s.T_ref - 3.0 * s.delta_T   # generous margin
            T_hi = s.T_ref + 4.0 * s.delta_T
            T_pred = np.clip(T_pred, T_lo, T_hi)
        
        info = {
            'wave_speed': output['wave_speed'].item(),
            'stable': output['stable'].item(),
            'hidden_state': output.get('hidden_state'),
        }
        
        return T_pred, info
    
    def fdm_step(self, T: np.ndarray, T_prev: np.ndarray,
                grid: np.ndarray, tau: float, num_steps: int = 1,
                k_temp_dependent: bool = False, 
                c_temp_dependent: bool = False,
                bc_left: float = None, bc_right: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform FDM step(s) using existing solver.
        
        Args:
            T: Current temperature
            T_prev: Previous temperature
            grid: Material grid
            tau: Relaxation time
            num_steps: Number of FDM steps
            k_temp_dependent: Temperature-dependent thermal conductivity
            c_temp_dependent: Temperature-dependent heat capacity
            bc_left: Left boundary condition temperature
            bc_right: Right boundary condition temperature
            
        Returns:
            T_new, info dict
        """
        TP = T.copy()
        TPP = T_prev.copy()
        
        bc_kw = {}
        if bc_left is not None and bc_right is not None:
            bc_kw['BC'] = (bc_left, bc_right)
        
        for _ in range(num_steps):
            T_new, _, info = nl_solve_HF_1d_Cattaneo(
                self.grid_size, grid, TP, TPP,
                dx=self.dx, dt=self.dt, tau=tau,
                tol=1e-6, max_newton_iters=20,
                gmres_maxit=200,
                k_temp_dependent=k_temp_dependent,
                c_temp_dependent=c_temp_dependent,
                verbose=False,
                solver_backend=self.config.fdm_backend,
                **bc_kw
            )
            TPP = TP.copy()
            TP = T_new
        
        return TP, info
    
    def adapt_timestep_jump(self, error: float):
        """
        Adapt timestep jump based on error history.
        
        Args:
            error: Current error estimate
        """
        self.error_history.append(error)

        if not self.config.enable_adaptive_jump:
            return
        
        # Adapt based on recent errors
        if len(self.error_history) >= 3:
            avg_error = np.mean(list(self.error_history)[-3:])
            
            if avg_error < self.config.residual_threshold / 10:
                # Error very small, can increase jump conservatively
                new_jump = min(self.current_jump + 1, self.config.max_timestep_jump)
                if new_jump > self.current_jump:
                    self.current_jump = new_jump
                    if self.config.verbose:
                        print(f"  Increased jump to {self.current_jump}")
            elif avg_error > self.config.residual_threshold:
                # Error too large, decrease jump
                self.current_jump = max(self.current_jump - 2, 1)
                if self.config.verbose:
                    print(f"  Decreased jump to {self.current_jump}")
    
    def step(self, T: np.ndarray, T_prev: np.ndarray,
            grid: np.ndarray, q: np.ndarray, tau: float,
            alpha: np.ndarray, rho_cp: np.ndarray,
            bc_left: float, bc_right: float,
            k_temp_dependent: bool = False,
            c_temp_dependent: bool = False,
            hidden_state=None) -> Tuple[np.ndarray, str, Dict]:
        """
        Single hybrid step: neural with FDM fallback.
        
        Args:
            T: Current temperature
            T_prev: Previous temperature
            grid: Material grid
            q: Heat source
            tau: Relaxation time
            alpha: Thermal diffusivity
            rho_cp: ρ*c_p
            bc_left, bc_right: Boundary conditions
            k_temp_dependent: Temperature-dependent k
            c_temp_dependent: Temperature-dependent c
            hidden_state: recurrent memory tensor or None
            
        Returns:
            T_new, method_used ('neural' or 'fdm'), info
            (info['hidden_state'] carries the new memory when method is 'neural')
        """
        self.stats['total_steps'] += 1
        
        # Check if neural is available
        if self.model is None:
            # Fall back to FDM
            T_new, info = self.fdm_step(T, T_prev, grid, tau, 
                                       self.current_jump,
                                       k_temp_dependent, c_temp_dependent,
                                       bc_left, bc_right)
            self.stats['fdm_calls'] += 1
            return T_new, 'fdm', info
        
        # Check stability condition
        if not self.check_stability(T, tau, alpha):
            # Unstable for large step, use FDM with small steps
            if self.config.verbose:
                print("Stability check failed, using FDM")
            T_new, info = self.fdm_step(T, T_prev, grid, tau, 1,
                                       k_temp_dependent, c_temp_dependent,
                                       bc_left, bc_right)
            self.stats['fdm_calls'] += 1
            return T_new, 'fdm', info
        
        # Check gradient sharpness (shock detection)
        sharpness = self.compute_gradient_sharpness(T)
        if sharpness > self.config.gradient_threshold:
            # Sharp gradient detected, use FDM
            if self.config.verbose:
                print(f"Sharp gradient detected ({sharpness:.1f}), using FDM")
            T_new, info = self.fdm_step(T, T_prev, grid, tau, 1,
                                       k_temp_dependent, c_temp_dependent,
                                       bc_left, bc_right)
            self.stats['fdm_calls'] += 1
            return T_new, 'fdm', info
        
        # Try neural prediction
        try:
            T_pred, neural_info = self.neural_predict(
                T, T_prev, q, tau, alpha, rho_cp, bc_left, bc_right,
                hidden_state=hidden_state,
            )

            if not neural_info.get('stable', True):
                if self.config.verbose:
                    print("Neural prediction failed internal stability check, falling back to FDM")
                T_new, info = self.fdm_step(T, T_prev, grid, tau, 1,
                                           k_temp_dependent, c_temp_dependent,
                                           bc_left, bc_right)
                self.stats['fdm_calls'] += 1
                return T_new, 'fdm', info
            
            # Estimate residual
            residual = self.estimate_residual(
                T_pred, T, T_prev, q, tau, alpha, rho_cp
            )
            
            # Sanity check: reject if the dimensionless change is absurdly large
            if self.model is not None and hasattr(self.model, 'scaler'):
                delta_T_star = np.max(np.abs(T_pred - T)) / self.model.scaler.delta_T
            else:
                delta_T_star = np.max(np.abs(T_pred - T)) / (np.ptp(T) + 1e-10)
            # A single step should change T* by O(dt_star) at most a few units
            max_plausible_change = 5.0   # dimensionless
            if delta_T_star > max_plausible_change:
                if self.config.verbose:
                    print(f"Neural change too large (ΔT*={delta_T_star:.2f}), falling back to FDM")
                if self.config.fallback_on_failure:
                    self.stats['fallbacks'] += 1
                    T_new, info = self.fdm_step(T, T_prev, grid, tau, 1,
                                               k_temp_dependent, c_temp_dependent,
                                               bc_left, bc_right)
                    self.stats['fdm_calls'] += 1
                    self.current_jump = max(self.current_jump - 2, 1)
                    return T_new, 'fdm', info
            
            # Check if residual is acceptable
            if residual < self.config.residual_threshold:
                if np.allclose(q, 0.0) and self.violates_relaxation(
                        T_pred, T, bc_left, bc_right):
                    T_pred = self.project_relaxing_step(T_pred, T, bc_left, bc_right)

                # Neural prediction accepted
                self.stats['neural_calls'] += 1
                self.adapt_timestep_jump(residual)
                
                info = {
                    'residual': residual,
                    'wave_speed': neural_info['wave_speed'],
                    'timestep_jump': self.current_jump
                }
                
                return T_pred, 'neural', info
            else:
                # Residual too high, fall back to FDM
                if self.config.verbose:
                    print(f"Residual too high ({residual:.2e}), falling back to FDM")
                
                if self.config.fallback_on_failure:
                    self.stats['fallbacks'] += 1
                    T_new, info = self.fdm_step(T, T_prev, grid, tau, 1,
                                               k_temp_dependent, c_temp_dependent,
                                               bc_left, bc_right)
                    self.stats['fdm_calls'] += 1
                    
                    # Adapt jump downward
                    self.current_jump = max(self.current_jump - 2, 1)
                    
                    return T_new, 'fdm', info
                else:
                    # Use neural anyway
                    self.stats['neural_calls'] += 1
                    return T_pred, 'neural', {'residual': residual, 'forced': True}
        
        except Exception as e:
            # Neural prediction failed, fall back to FDM
            if self.config.verbose:
                print(f"Neural prediction failed: {e}, falling back to FDM")
            
            self.stats['fallbacks'] += 1
            T_new, info = self.fdm_step(T, T_prev, grid, tau, 1,
                                       k_temp_dependent, c_temp_dependent,
                                       bc_left, bc_right)
            self.stats['fdm_calls'] += 1
            
            return T_new, 'fdm', info
    
    def solve(self, T0: np.ndarray, grid: np.ndarray, 
             num_steps: int, tau: float, alpha: np.ndarray,
             rho_cp: np.ndarray, q_func: Optional[Callable] = None,
             bc_left: float = 10.0, bc_right: float = 50.0,
             k_temp_dependent: bool = False,
             c_temp_dependent: bool = False,
             save_history: bool = False) -> Dict:
        """
        Solve Cattaneo equation using hybrid approach.
        
        Args:
            T0: Initial temperature
            grid: Material grid
            num_steps: Number of timesteps
            tau: Relaxation time
            alpha: Thermal diffusivity
            rho_cp: ρ*c_p
            q_func: Optional heat source function q(t, x)
            bc_left, bc_right: Boundary conditions
            k_temp_dependent: Temperature-dependent k
            c_temp_dependent: Temperature-dependent c
            save_history: Whether to save full history
            
        Returns:
            dict with solution and statistics
        """
        # Initialize
        T = T0.copy()
        T_prev = T0.copy()
        
        # Storage
        T_history = [T.copy()] if save_history else None
        method_history = []
        neural_count = 0  # running count for O(1) percentage computation
        total_effective_fdm_steps = 0  # Track total physical time in FDM-equivalent steps
        hidden_state = None  # recurrent memory
        
        for step in range(num_steps):
            # Compute heat source
            if q_func is not None:
                q = q_func(step * self.dt, self.dx * np.arange(self.grid_size))
            else:
                q = np.zeros(self.grid_size)
            
            # Hybrid step
            T_new, method, info = self.step(
                T, T_prev, grid, q, tau, alpha, rho_cp,
                bc_left, bc_right, k_temp_dependent, c_temp_dependent,
                hidden_state=hidden_state,
            )
            # Carry recurrent memory forward (only updated on neural steps)
            if method == 'neural':
                hidden_state = info.get('hidden_state')
            
            # Track effective timestep advancement
            if method == 'neural':
                total_effective_fdm_steps += self.current_jump
            else:
                total_effective_fdm_steps += 1
            
            # Update
            T_prev = T.copy()
            T = T_new
            
            # Store
            if save_history:
                T_history.append(T.copy())
            method_history.append(method)
            if method == 'neural':
                neural_count += 1

            if self.config.verbose and step % 100 == 0:
                neural_pct = 100 * neural_count / (step + 1)
                print(f"Step {step}/{num_steps}: {neural_pct:.1f}% neural")

        result = {
            'T_final': T,
            'stats': self.stats.copy(),
            'neural_percentage': 100 * neural_count / max(num_steps, 1),
            'method_history': method_history,
            'total_effective_fdm_steps': total_effective_fdm_steps
        }
        
        if save_history:
            result['T_history'] = np.array(T_history)
        
        return result
    
    def print_stats(self):
        """Print solver statistics."""
        print("\nHybrid Solver Statistics:")
        print(f"  Total steps: {self.stats['total_steps']}")
        print(f"  Neural calls: {self.stats['neural_calls']}")
        print(f"  FDM calls: {self.stats['fdm_calls']}")
        print(f"  Fallbacks: {self.stats['fallbacks']}")
        if self.stats['total_steps'] > 0:
            neural_pct = 100 * self.stats['neural_calls'] / self.stats['total_steps']
            print(f"  Neural percentage: {neural_pct:.1f}%")


if __name__ == '__main__':
    # Test hybrid solver
    grid_size = 112
    dx = 1e-8
    dt = 1e-13
    tau = 1e-9
    
    # Initialize
    T0 = np.ones(grid_size) * 10.0
    grid = np.ones(grid_size, dtype=int)
    alpha = np.ones(grid_size) * 1e-4
    rho_cp = np.ones(grid_size) * 1e6
    
    # Create solver without model (FDM only)
    config = HybridSolverConfig(verbose=True)
    solver = HybridCattaneoSolver(grid_size, dx, dt, config=config)
    
    # Solve
    result = solver.solve(
        T0, grid, num_steps=10, tau=tau,
        alpha=alpha, rho_cp=rho_cp,
        bc_left=10.0, bc_right=50.0,
        save_history=True
    )
    
    solver.print_stats()
    print(f"\nFinal temperature range: [{result['T_final'].min():.2f}, {result['T_final'].max():.2f}]")
