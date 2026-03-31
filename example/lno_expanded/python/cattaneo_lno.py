"""
Cattaneo-LNO: Laplace Neural Operator for Hyperbolic Heat Conduction
=====================================================================

Implements a specialized LNO-style architecture for the dimensionless Cattaneo equation.
The Laplace Neural Operator replaces the Fourier-based spectral convolution with a
learned kernel in the Laplace domain, using exponentially-decaying basis functions
that are naturally suited to dissipative/hyperbolic PDEs.

Nondimensionalization (spatially-varying Fo/Ve formulation):
    Each training sample has its own *spatially varying* material
    properties α(x) and τ(x), so the dimensionless groups are
    computed **per grid point**:

    Fourier number   Fo(x) = α(x) · dt / dx²    (local diffusive CFL)
    Vernotte number  Ve(x) = √(α(x) · τ(x)) / L  (local wave penetration)

    The Cattaneo residual in discrete weak form becomes:
        (Ve(x)² / Fo(x)²) · (T*ⁿ⁺¹ − 2T*ⁿ + T*ⁿ⁻¹)
      + (1  / Fo(x) ) · (T*ⁿ⁺¹ − T*ⁿ)
      − lap(T*ⁿ⁺¹)
      − q*
      = 0

    where lap(T*) = (T*_{i+1} − 2T*_i + T*_{i-1}) / dx*²,
    dx* = dx / L, and q* = q / q_c with q_c = α_ref · ΔT / L².

    The network input channels are 7:
        [T*_n, T*_{n-1}, Fo(x), Ve(x), x_coord, bc_left*, bc_right*]

    Fo is clamped to [1e-6, 1.0] per grid point to prevent extreme
    Ve²/Fo² values.

Key design choices:
- **Spatially varying Fo/Ve**: Both the model input features and the
  physics loss use Fo(x) and Ve(x) computed from the actual per-point
  α(x) and τ(x), not from global reference values or spatial means.
  This ensures the physics residual coefficients match the material
  properties used to generate each training sample at every grid point.
- **Structural hard BC constraint**: Dirichlet BCs are built into the
  output function via a lift–distance decomposition
  T*_{n+1} = T*_n + G_inc(ξ) + D(ξ)·N(ξ), where D(ξ)=ξ(1−ξ) vanishes
  at boundaries and G_inc linearly interpolates the required boundary
  increments.  Unlike post-hoc clamping, this preserves gradient flow at
  boundary-adjacent nodes and prevents autoregressive drift.
- **Ghost boundary cells**: Two ghost cells carry Dirichlet BC values,
  providing stencil context to the LNO backbone.
- **Transient-optimized filter**: Preserves 80% of spectral modes at full
  strength, tapering only the extreme tail for sharp thermal fronts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math


class DimensionlessScaler:
    """
    Handles conversion between dimensional and dimensionless variables
    using the Fourier number (Fo) and Vernotte number (Ve).

    Fourier number (diffusive CFL):
        Fo = α · dt / dx²

    Vernotte number (wave-to-domain ratio):
        Ve = √(α · τ) / L

    Temperature scaling:
        T* = (T − T_ref) / ΔT

    Spatial scaling:
        x* = x / L
        dx* = dx / L

    The correct Cattaneo residual (dimensional PDE ÷ ΔT/dt):
        (τ/dt)·d2T* + dT* = Fo · sec_diff(T*)
    where sec_diff = T*_{j+1} − 2T*_j + T*_{j-1} (NOT divided by dx*²).

    Ve is used as a network input feature but does NOT appear in the
    physics loss coefficients.  The physics loss uses τ/dt and Fo directly.

    Fo is clamped to [Fo_min, Fo_max] for the network input features.
    """
    Fo_min = 1e-6   # lower clamp for Fo
    Fo_max = 500.0  # upper clamp (supports large timestep_jump)

    def __init__(self, L: float = 1.0, alpha_ref: float = 1e-4,
                 T_ref: float = 200.0, delta_T: float = 100.0,
                 tau_ref: float = 1e-9):
        """
        Args:
            L: Domain length [m]
            alpha_ref: Reference thermal diffusivity [m²/s]
            T_ref: Reference temperature [K]
            delta_T: Characteristic temperature difference [K]
            tau_ref: Reference relaxation time [s]
        """
        self.L = L
        self.alpha_ref = alpha_ref
        self.T_ref = T_ref
        self.delta_T = delta_T
        self.tau_ref = tau_ref

        # Derived scales
        self.t_c = L**2 / alpha_ref                      # characteristic time [s]
        self.c_ref = math.sqrt(alpha_ref / tau_ref)     # reference wave speed [m/s]
        self.q_c = alpha_ref * delta_T / L**2            # heat-source scale

    # ── Fo / Ve computation ──

    def compute_Fo(self, alpha: float, dt: float, dx: float) -> float:
        """Fourier number: Fo = α · dt / dx²  (clamped to [Fo_min, Fo_max])."""
        Fo = alpha * dt / (dx**2)
        return max(self.Fo_min, min(Fo, self.Fo_max))

    def compute_Ve(self, alpha: float, tau: float, L: float) -> float:
        """Vernotte number: Ve = √(α · τ) / L."""
        return math.sqrt(alpha * tau) / L

    @staticmethod
    def compute_Fo_batch(alpha: torch.Tensor, dt: float, dx: float,
                         Fo_min: float = 1e-6,
                         Fo_max: float = 1.0) -> torch.Tensor:
        """Vectorised Fourier number for batched alpha (spatial mean).

        Args:
            alpha: Per-sample diffusivity [batch] or [batch, grid]
            dt: Time step (scalar)
            dx: Spatial step (scalar)
            Fo_min: Lower clamp

        Returns:
            Fo: [batch] tensor, clamped to [Fo_min, Fo_max]
        """
        if alpha.dim() == 2:
            alpha = alpha.mean(dim=-1)  # spatial mean → [batch]
        Fo = alpha * dt / (dx ** 2)
        return torch.clamp(Fo, min=Fo_min, max=Fo_max)

    @staticmethod
    def compute_Ve_batch(alpha: torch.Tensor, tau: torch.Tensor,
                         L: float) -> torch.Tensor:
        """Vectorised Vernotte number for batched alpha and tau (spatial mean).

        Args:
            alpha: Per-sample diffusivity [batch] or [batch, grid]
            tau: Per-sample relaxation time [batch] or [batch, grid]
            L: Domain length (scalar)

        Returns:
            Ve: [batch] tensor
        """
        if alpha.dim() == 2:
            alpha = alpha.mean(dim=-1)
        if tau.dim() == 2:
            tau = tau.mean(dim=-1)
        return torch.sqrt(alpha * tau) / L

    @staticmethod
    def compute_Fo_field(alpha: torch.Tensor, dt: float, dx: float,
                         Fo_min: float = 1e-6,
                         Fo_max: float = 1.0) -> torch.Tensor:
        """Spatially varying Fourier number field.

        Preserves the spatial dimension — no averaging.

        Args:
            alpha: [batch, grid] thermal diffusivity field
            dt: Time step (scalar)
            dx: Spatial step (scalar)
            Fo_min: Lower clamp

        Returns:
            Fo: [batch, grid] tensor, clamped to [Fo_min, Fo_max]
        """
        Fo = alpha * dt / (dx ** 2)
        return torch.clamp(Fo, min=Fo_min, max=Fo_max)

    @staticmethod
    def compute_Ve_field(alpha: torch.Tensor, tau: torch.Tensor,
                         L: float) -> torch.Tensor:
        """Spatially varying Vernotte number field.

        Preserves the spatial dimension — no averaging.

        Args:
            alpha: [batch, grid] thermal diffusivity field
            tau: [batch, grid] relaxation time field
            L: Domain length (scalar)

        Returns:
            Ve: [batch, grid] tensor
        """
        return torch.sqrt(alpha * tau) / L

    # ── Temperature scaling ──

    def to_dimensionless_temp(self, T: torch.Tensor) -> torch.Tensor:
        """T* = (T − T_ref) / ΔT"""
        return (T - self.T_ref) / self.delta_T

    def from_dimensionless_temp(self, T_star: torch.Tensor) -> torch.Tensor:
        """T = T_ref + T* · ΔT"""
        return self.T_ref + T_star * self.delta_T

    # ── Material-parameter scaling ──

    def to_dimensionless_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        """α* = α / α_ref"""
        return alpha / self.alpha_ref

    def to_dimensionless_q(self, q: torch.Tensor) -> torch.Tensor:
        """Q* = q / q_c"""
        return q / self.q_c

    # ── Spatial scaling ──

    def get_dx_star(self, dx: float) -> float:
        """dx* = dx / L"""
        return dx / self.L
    def get_dt_star(self, dt:float) -> float:
        """dt* = dt / t_c"""
        return dt / self.t_c


class LaplaceConv1d(nn.Module):
    """
    Laplace Neural Operator convolution layer.

    Instead of Fourier modes (oscillatory basis), this layer uses a set of
    learned exponentially-decaying Laplace-domain poles {s_k} to build the
    integral kernel.  Each pole contributes an exponential basis function
    exp(-s_k |x-y|) weighted by a learned complex amplitude (polar form),
    and the kernel is the superposition of these contributions.

    This is well-suited for dissipative and hyperbolic PDEs (like the
    Cattaneo equation) where the Green's function decays exponentially.

    The `modes` parameter controls the number of Laplace poles (analogous
    to the number of Fourier modes in an FNO).

    Key features:
    - Polar weight parameterisation: amplitude is bounded via sigmoid,
      phase is free — prevents unbounded spectral energy growth.
    - Data-dependent poles: a small MLP shifts pole locations based on
      the input signal, allowing the kernel shape to adapt per-sample.
    - Data-dependent causal mask: uses per-sample c* = Ve/√Fo from the
      physics to mask non-causal interactions in the kernel.
    - Configurable anti-aliasing filter (dealias / exponential / etc.).
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int,
                 causal_mask: bool = True, wave_speed: Optional[float] = None,
                 spectral_filter: str = 'exponential',
                 filter_strength: float = 4.0,
                 max_amp: float = 1.0,
                 amp_sharpness: float = 1.0,
                 pole_offset_scale: float = 0.1,
                 pole_min: float = 0.1,
                 pole_max: float = 100.0,
                 use_causal_mask: bool = True,
                 causal_safety: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.causal_mask = causal_mask
        self.wave_speed = wave_speed  # legacy static fallback
        self.spectral_filter = spectral_filter
        self.filter_strength = filter_strength

        # ── Polar weight parameterisation ──
        # Amplitude is bounded via sigmoid to prevent unbounded spectral
        # energy growth during long autoregressive rollouts.
        self.max_amp = max_amp
        self.amp_sharpness = amp_sharpness

        # Kaiming-correct init for the polar parameterisation.
        # The output einsum contracts over (modes, in_channels), producing
        # out[b,o,s] = sum_k sum_i x_k[k,b,i,s] * amp[k,i,o] * cos(phase[k,i,o])
        # For Var(out) = Var(x) we need:
        #   modes * in_channels * E[amp²] * E[cos²(phase)] = 1
        #   E[cos²] = 0.5, so E[amp²] = 2 / (modes * in_channels)
        #   amp_target = sqrt(2 / (modes * in_channels))
        # Since amp = max_amp * sigmoid(sharpness * log_amp), we bias
        # log_amp so sigmoid gives the target amplitude at init.
        _fan_in = modes * in_channels
        _target_amp = math.sqrt(2.0 / _fan_in)
        _target_sigmoid = max(0.01, min(0.99, _target_amp / max_amp))
        _init_bias = math.log(_target_sigmoid / (1.0 - _target_sigmoid)) / amp_sharpness
        init_scale = 1.0 / (in_channels * out_channels * modes) ** 0.5
        self.weight_log_amp = nn.Parameter(
            _init_bias + torch.randn(modes, in_channels, out_channels) * init_scale
        )
        self.weight_phase = nn.Parameter(
            torch.randn(modes, in_channels, out_channels)
        )

        # ── Learnable Laplace poles (positive real parts for stability) ──
        # Poles operate on normalised [0,1] distances.  A pole of ~1 gives
        # domain-wide interaction (exp(-1)≈0.37 at boundary), while ~30
        # gives ultra-local interaction.  Initialise as a log-spaced
        # range so that the model starts with a good multi-scale basis.
        log_poles_init = torch.linspace(
            math.log(1.0), math.log(50.0), modes
        )
        self.log_poles = nn.Parameter(log_poles_init)

        # ── Data-dependent pole offsets ──
        self.pole_mlp = nn.Sequential(
            nn.Linear(in_channels, modes),
            nn.Tanh()
        )
        self.pole_offset_scale = pole_offset_scale
        self.pole_min = pole_min
        self.pole_max = pole_max

        # ── Causal mask config ──
        self.use_causal_mask = use_causal_mask
        self.causal_safety = causal_safety

        if spectral_filter == 'learnable':
            self.filter_alpha = nn.Parameter(torch.tensor(filter_strength))

        # Lazy caches
        self._cached_dist_size = None
        self._cached_dist = None
        self._cached_filter_n = None
        self._cached_filter = None

    def _build_spectral_filter(self, n_modes_used: int,
                                device: torch.device) -> torch.Tensor:
        """
        Build a filter over the Laplace poles.

        Higher-index poles correspond to faster-decaying (more local)
        basis functions.  The filter can attenuate contributions from
        specific poles to regularise the operator.
        """
        if self.spectral_filter == 'none' or n_modes_used <= 1:
            return torch.ones(n_modes_used, device=device)

        k = torch.arange(n_modes_used, dtype=torch.float32, device=device)
        k_norm = k / max(n_modes_used - 1, 1)

        if self.spectral_filter == 'exponential':
            sigma = self.filter_strength
            filt = torch.exp(-sigma * k_norm ** 2)
        elif self.spectral_filter == 'raised_cosine':
            filt = 0.5 * (1.0 + torch.cos(math.pi * k_norm))
        elif self.spectral_filter == 'learnable':
            sigma = F.softplus(self.filter_alpha)
            filt = torch.exp(-sigma * k_norm ** 2)
        elif self.spectral_filter == 'sharp_cutoff':
            cutoff = int(n_modes_used * 2 / 3)
            filt = torch.ones(n_modes_used, device=device)
            filt[cutoff:] = 0.0
        elif self.spectral_filter == 'dealias':
            # Strong 2/3-rule de-aliasing: zero out upper third of modes
            cutoff = int(n_modes_used * self.filter_strength)
            filt = torch.ones(n_modes_used, device=device)
            filt[cutoff:] = 0.0
        elif self.spectral_filter == 'transient_optimized':
            cutoff = int(n_modes_used * 0.8)
            filt = torch.ones(n_modes_used, device=device)
            if n_modes_used - cutoff > 0:
                filt[cutoff:] = torch.linspace(1.0, 0.0, n_modes_used - cutoff,
                                                device=device)
        else:
            filt = torch.ones(n_modes_used, device=device)

        return filt

    def forward(self, x: torch.Tensor, dt_star: Optional[float] = None,
                dx_star: Optional[float] = None,
                c_star: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Laplace-domain integral operator on a 1D grid.

        For each Laplace pole s_k the kernel contribution is:
            K_k(x_i, x_j) = exp(-s_k * |x_i - x_j|)
        The output is:
            u_out(x_i) = sum_k  filter_k * W_k @ [ sum_j K_k(x_i,x_j) u_in(x_j) ]

        Args:
            x: [batch, channels, grid]
            dt_star: dimensionless time step (scalar)
            dx_star: dimensionless spatial step (scalar)
            c_star: per-sample dimensionless wave speed [batch] or [batch, grid]
        """
        batch_size, _, size = x.shape
        device = x.device

        # ── Data-dependent poles ──
        # x_mean: global spatial-channel mean → [in_channels]
        x_mean = x.mean(dim=(0, 2))  # [in_channels]  (fused reduction)
        pole_offsets = self.pole_mlp(x_mean)  # [modes]
        poles = F.softplus(self.log_poles + self.pole_offset_scale * pole_offsets)
        poles = torch.clamp(poles, min=self.pole_min, max=self.pole_max)

        n_poles = min(self.modes, max(1, size))

        # ── Polar weights → real mixing matrix ──
        weight_amp = torch.sigmoid(self.amp_sharpness * self.weight_log_amp)
        weight_amp = self.max_amp * weight_amp  # bounded ∈ [0, max_amp]
        weights = weight_amp * torch.cos(self.weight_phase)  # real projection

        # ── Pole-wise filter ──
        if self.spectral_filter != 'learnable':
            if (self._cached_filter_n != n_poles or self._cached_filter is None
                    or self._cached_filter.device != device):
                self._cached_filter = self._build_spectral_filter(n_poles, device)
                self._cached_filter_n = n_poles
            pole_filter = self._cached_filter
        else:
            pole_filter = self._build_spectral_filter(n_poles, device)

        # ── Pairwise absolute distances ──
        if (self._cached_dist_size != size or self._cached_dist is None
                or self._cached_dist.device != device):
            coords = torch.linspace(0.0, 1.0, size, device=device)
            self._cached_dist = (coords.unsqueeze(0) - coords.unsqueeze(1)).abs()
            self._cached_dist_size = size
        dist = self._cached_dist

        # ── Data-dependent causal mask ──
        if (self.use_causal_mask and c_star is not None
                and dt_star is not None and dx_star is not None):
            # Conservative bound: use max c* across the batch.
            # c_star and dt_star are both in consistent nondimensional units
            # (domain [0,1], time scale t_c = L²/α) so their product is
            # directly in the [0,1] coordinate range used by `dist`.
            c_eff = self.causal_safety * c_star.max()
            max_dist_norm = c_eff * dt_star
            causal = (dist <= max_dist_norm).float()
        elif (self.causal_mask and self.wave_speed is not None
                and dt_star is not None and dx_star is not None):
            # Legacy static fallback
            max_dist = max(1e-12, self.wave_speed * dt_star / (dx_star + 1e-12))
            max_dist_norm = max_dist / max(size - 1, 1)
            causal = (dist <= max_dist_norm).float()
        else:
            causal = torch.ones_like(dist)

        # ── Build kernel matrices ──  [n_poles, size, size]
        # Poles act on normalised [0,1] coordinates so that a pole value of
        # ~1 gives domain-wide interaction and ~10 gives local interaction.
        # (Do NOT multiply by `size` — that would make every kernel ultra-
        # local and kill gradient flow through the pole parameters.)
        #
        # We normalise by the **detached** row-sum so the scale stays O(1)
        # but gradients still flow freely through the exponential kernel
        # to the pole parameters.  Standard row-normalisation creates a
        # softmax-like cancellation that drives ∂kernel/∂pole → 0.
        raw_kernels = torch.exp(
            -poles[:n_poles, None, None] * dist.unsqueeze(0)
        ) * causal.unsqueeze(0)
        row_norms = raw_kernels.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        kernels = raw_kernels / row_norms.detach()  # stop-gradient normalisation
        kernels = kernels * pole_filter[:, None, None]

        # ── Fused batched kernel application + weight mixing ──
        out = torch.einsum('bcs,kts,kio->bot', x, kernels, weights[:n_poles])

        return out


class CausalLNOBlock(nn.Module):
    """LNO block with characteristic-aware processing.

    Residual structure:  pointwise → activation → spectral → add residual.
    The spectral operator is NOT bypassed — information must flow through
    the Laplace kernel, preventing the network from learning a trivial
    identity shortcut around the operator.
    """
    def __init__(self, channels: int, modes: int,
                 activation: str = 'swish', causal_mask: bool = True,
                 wave_speed: float = 1.0,
                 spectral_filter: str = 'exponential',
                 filter_strength: float = 4.0,
                 use_spectral_norm: bool = False,
                 max_amp: float = 1.0,
                 amp_sharpness: float = 1.0,
                 pole_offset_scale: float = 0.1,
                 pole_min: float = 0.1,
                 pole_max: float = 100.0,
                 use_causal_mask: bool = True,
                 causal_safety: float = 1.0):
        super().__init__()
        self.channels = channels
        self.modes = modes

        # Laplace neural integral operator
        self.spectral_conv = LaplaceConv1d(
            channels, channels, modes, causal_mask,
            wave_speed=wave_speed,
            spectral_filter=spectral_filter,
            filter_strength=filter_strength,
            max_amp=max_amp,
            amp_sharpness=amp_sharpness,
            pole_offset_scale=pole_offset_scale,
            pole_min=pole_min,
            pole_max=pole_max,
            use_causal_mask=use_causal_mask,
            causal_safety=causal_safety,
        )

        # Pointwise (1x1) convolution for local mixing.
        _pointwise = nn.Conv1d(channels, channels, 1)
        self.pointwise = (nn.utils.spectral_norm(_pointwise)
                          if use_spectral_norm else _pointwise)

        # Activation
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, dt_star: Optional[float] = None,
                dx_star: Optional[float] = None,
                c_star: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with residual connection.

        Residual structure: pointwise → activate → spectral → add residual.
        No bypass around the spectral operator.
        """
        residual = x

        # Pointwise mixing then activation BEFORE spectral
        x = self.pointwise(x)
        x = self.activation(x)

        # Spectral path (with data-dependent causal mask)
        x = self.spectral_conv(x, dt_star, dx_star, c_star)

        # Residual connection
        return x + residual


class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder for wave-like behavior."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        self.projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequence."""
        _, hidden = self.gru(x_seq)
        encoded = hidden[-1]
        encoded = self.projection(encoded)
        return encoded


class TemporalConvEncoder(nn.Module):
    """Causal temporal convolution encoder for mandatory history processing.

    Replaces the optional GRU with causal 1-D convolutions along the time
    axis.  Always runs — the model never skips temporal encoding.

    Input:  [batch, K, grid]  (K history frames, oldest first)
    Output: [batch, out_channels, grid]
    """
    def __init__(self, history_len: int, out_channels: int,
                 mid_channels: int = 32):
        super().__init__()
        self.history_len = history_len
        # Encode K frames → mid → out via depthwise-separable style
        self.net = nn.Sequential(
            # Treat time axis as channel dim: [B, K, G] → [B, mid, G]
            nn.Conv1d(history_len, mid_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1),
            nn.SiLU(),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: [batch, K, grid] → [batch, out_channels, grid]"""
        return self.net(history)


class LocalConvPath(nn.Module):
    """Dilated 1-D convolution stack for local spatial gradients.

    Captures sharp boundary layers and near-boundary behavior that the
    global Laplace path may blur.
    """
    def __init__(self, channels: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = dilation  # same-length output
            layers.append(nn.Conv1d(channels, channels, kernel_size=3,
                                    padding=padding, dilation=dilation))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, grid] → same shape."""
        return self.net(x)


class Conv1dGRUCell(nn.Module):
    """Spatial GRU cell operating pointwise along the grid via 1x1 convolutions.

    Maintains a per-spatial-point hidden state ``[B, hidden, G]`` that
    persists across autoregressive time-steps, giving the model a
    *recurrent memory* of its own trajectory history beyond the fixed-
    length history window.

    All gates use ``kernel_size=1`` so information is propagated
    spatially only through the upstream backbone blocks; the GRU's
    role is purely temporal aggregation at each grid point.
    """

    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        # Gates: reset r, update z
        self.W_rz = nn.Conv1d(input_channels + hidden_channels,
                              2 * hidden_channels, kernel_size=1)
        # Candidate hidden state
        self.W_h = nn.Conv1d(input_channels + hidden_channels,
                             hidden_channels, kernel_size=1)

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features  ``[B, input_channels, G]``
            h: previous hidden ``[B, hidden_channels, G]`` or *None* (zero-init)

        Returns:
            h_new: ``[B, hidden_channels, G]``
        """
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2],
                            device=x.device, dtype=x.dtype)
        xh = torch.cat([x, h], dim=1)
        rz = torch.sigmoid(self.W_rz(xh))
        r, z = rz.chunk(2, dim=1)
        xrh = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.W_h(xrh))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class MemoryFusion(nn.Module):
    """Fuse recurrent memory into backbone features via gated addition.

    Produces a gate ``g ∈ [0, 1]`` per channel and grid point and
    injects the memory as ``h_out = h_backbone + g * proj(memory)``.
    """

    def __init__(self, backbone_channels: int, memory_channels: int):
        super().__init__()
        self.proj = nn.Conv1d(memory_channels, backbone_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv1d(backbone_channels + memory_channels, backbone_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_backbone: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_backbone: ``[B, C, G]``
            memory: ``[B, M, G]``
        Returns:
            ``[B, C, G]`` — backbone features enriched with memory.
        """
        g = self.gate(torch.cat([h_backbone, memory], dim=1))
        return h_backbone + g * self.proj(memory)


class MultiScaleLNOBlock(nn.Module):
    """Multi-scale block: global Laplace path + local dilated-conv path.

    Residual structure per block:
        h_{k+1} = h_k + F_global(h_k + t_k) + F_local(h_k + t_k)

    where t_k is the per-block temporal injection.
    """
    def __init__(self, channels: int, modes: int,
                 temporal_channels: int,
                 activation: str = 'swish',
                 local_layers: int = 2,
                 causal_mask: bool = True,
                 wave_speed: float = 1.0,
                 spectral_filter: str = 'exponential',
                 filter_strength: float = 4.0,
                 use_spectral_norm: bool = False,
                 max_amp: float = 1.0,
                 amp_sharpness: float = 1.0,
                 pole_offset_scale: float = 0.1,
                 pole_min: float = 0.1,
                 pole_max: float = 100.0,
                 use_causal_mask: bool = True,
                 causal_safety: float = 1.0):
        super().__init__()

        # ── Global Laplace path ──
        _pointwise = nn.Conv1d(channels, channels, 1)
        self.pointwise = (nn.utils.spectral_norm(_pointwise)
                          if use_spectral_norm else _pointwise)
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()
        self.spectral_conv = LaplaceConv1d(
            channels, channels, modes, causal_mask,
            wave_speed=wave_speed,
            spectral_filter=spectral_filter,
            filter_strength=filter_strength,
            max_amp=max_amp,
            amp_sharpness=amp_sharpness,
            pole_offset_scale=pole_offset_scale,
            pole_min=pole_min,
            pole_max=pole_max,
            use_causal_mask=use_causal_mask,
            causal_safety=causal_safety,
        )

        # ── Local dilated-conv path ──
        self.local_path = LocalConvPath(channels, num_layers=local_layers)

        # ── Per-block temporal fusion (project temporal features → channels) ──
        self.temporal_gate = nn.Sequential(
            nn.Conv1d(temporal_channels, channels, 1),
            nn.Sigmoid(),
        )
        self.temporal_proj = nn.Conv1d(temporal_channels, channels, 1)

    def forward(self, x: torch.Tensor,
                temporal_features: torch.Tensor,
                dt_star: Optional[float] = None,
                dx_star: Optional[float] = None,
                c_star: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, grid]
            temporal_features: [batch, temporal_channels, grid]
        """
        residual = x

        # Inject temporal context via gated addition
        t_gate = self.temporal_gate(temporal_features)
        t_proj = self.temporal_proj(temporal_features)
        x = x + t_gate * t_proj

        # Global path: pointwise → activate → Laplace spectral
        x_global = self.pointwise(x)
        x_global = self.activation(x_global)
        x_global = self.spectral_conv(x_global, dt_star, dx_star, c_star)

        # Local path: dilated convolutions
        x_local = self.local_path(x)

        return residual + x_global + x_local


# ─── Differentiable batched tridiagonal solver (Thomas algorithm) ────
@torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
def tridiag_solve(lower: torch.Tensor, diag: torch.Tensor,
                  upper: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Batched tridiagonal solve via the Thomas algorithm.

    Solves  A·x = rhs  where A is tridiagonal with sub-diagonal *lower*,
    main diagonal *diag*, and super-diagonal *upper*.  Fully differentiable
    through autograd (no in-place ops).

    Args:
        lower: [B, N-1]  sub-diagonal  (row i couples to column i-1)
        diag:  [B, N]    main diagonal
        upper: [B, N-1]  super-diagonal (row i couples to column i+1)
        rhs:   [B, N]    right-hand side

    Returns:
        x: [B, N]  solution vector
    """
    B, N = diag.shape

    # Forward sweep
    c_list = []
    d_list = []
    c0 = upper[:, 0] / diag[:, 0]
    d0 = rhs[:, 0] / diag[:, 0]
    c_list.append(c0)
    d_list.append(d0)

    for i in range(1, N):
        denom = diag[:, i] - lower[:, i - 1] * c_list[i - 1]
        d_i = (rhs[:, i] - lower[:, i - 1] * d_list[i - 1]) / denom
        d_list.append(d_i)
        if i < N - 1:
            c_i = upper[:, i] / denom
            c_list.append(c_i)
        else:
            c_list.append(torch.zeros_like(d_i))

    # Back substitution
    x_list = [None] * N
    x_list[-1] = d_list[-1]
    for i in range(N - 2, -1, -1):
        x_list[i] = d_list[i] - c_list[i] * x_list[i + 1]

    return torch.stack(x_list, dim=1)


class WaveDiffusionHead(nn.Module):
    """Coefficient prediction head for structure-preserving update.

    Predicts spatially-varying coefficients aθ(x), bθ(x) such that:
        ΔT* = aθ · sec_diff(T*_n) + bθ · (T*_n − T*_{n-1})

    Physics-informed residual formulation:
        aθ = Fo/(1+τ/dt) + correction_a   (diffusion coefficient — learned)
        bθ = τ/dt/(1+τ/dt)                (wave memory — FIXED at physics target)

    The analytical targets Fo/(1+τ/dt) and τ/dt/(1+τ/dt) are computed from
    the input Fo/τ/dt fields.  The network learns corrections to aθ only.
    bθ is frozen at the physics target because:
      1. b_target < 1 guarantees rollout stability (wave memory damps)
      2. b_target is accurate to 0.02% vs FDM truth
      3. The loss landscape is degenerate (sec_diff << diff_T), so
         optimising b freely causes it to drift above 1.0 → divergence
    The IterativeCorrector handles any remaining discrepancy.

    MAX_CORRECTION_FRAC is kept very small (1%) because:
      - Physics targets match FDM to 0.02%, so large corrections are wrong.
      - At 25%, tanh saturation traps the correction at the bound with
        zero gradient, preventing recovery — a_theta stuck at 0.75×a_target.
      - At 1%, even full saturation gives ≤1% error; the corrector handles the rest.
    """
    # Maximum fractional deviation from physics target (for a only)
    MAX_CORRECTION_FRAC = 0.01

    def __init__(self, channels: int):
        super().__init__()
        # Correction network: predicts small residual for aθ only
        self.coeff_net = nn.Sequential(
            nn.Conv1d(channels + 2, channels, 1),  # +2 for Fo, Ve
            nn.SiLU(),
            nn.Conv1d(channels, channels // 2, 1),
            nn.SiLU(),
            nn.Conv1d(channels // 2, 1, 1),  # output: raw δaθ only
        )
        # Near-zero init so corrections start negligible
        nn.init.normal_(self.coeff_net[-1].weight, std=1e-3)
        nn.init.zeros_(self.coeff_net[-1].bias)

    @torch.compiler.disable
    def _physics_targets(self, Fo: torch.Tensor, tau_dt: torch.Tensor) -> torch.Tensor:
        """Analytical coefficient targets: aθ=Fo/(1+τ/dt), bθ=(τ/dt)/(1+τ/dt)."""
        denom = 1.0 + tau_dt                  # [B, G]
        a_target = Fo / denom
        b_target = tau_dt / denom
        return torch.stack([a_target, b_target], dim=1)  # [B, 2, G]

    def forward(self, h: torch.Tensor,
                Fo: torch.Tensor, Ve: torch.Tensor,
                tau_dt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h:      [batch, channels, grid]
            Fo:     [batch, grid]
            Ve:     [batch, grid]  — used as network feature
            tau_dt: [batch, grid]  — τ/dt, used for physics targets
        Returns:
            coeffs: [batch, 2, grid]  (channel 0 = aθ, channel 1 = bθ)
        """
        # ── Analytical physics targets (outside compile for inductor compat) ──
        if tau_dt is None:
            tau_dt = Ve  # backward compat fallback
        physics = self._physics_targets(Fo, tau_dt)  # [B, 2, G]

        # ── Learned correction for a only (b frozen at physics target) ──
        Fo_ch = torch.log1p(Fo).unsqueeze(1)    # [B, 1, G]
        Ve_ch = Ve.unsqueeze(1)                  # [B, 1, G]
        inp = torch.cat([h, Fo_ch, Ve_ch], dim=1)
        raw_a_correction = self.coeff_net(inp)   # [B, 1, G]

        # Bound a correction: tanh squashes to [-1, 1], scale by fraction of |a_target|
        a_target = physics[:, 0:1]               # [B, 1, G]
        bound_a = self.MAX_CORRECTION_FRAC * a_target.abs().clamp(min=1e-6)
        a_corrected = a_target + bound_a * torch.tanh(raw_a_correction)

        # b is fixed at physics target (guaranteed < 1 for rollout stability)
        b_fixed = physics[:, 1:2]                # [B, 1, G]

        return torch.cat([a_corrected, b_fixed], dim=1)  # [B, 2, G]


class IterativeCorrector(nn.Module):
    """Learned iterative correction gated by the physics signal magnitude.

    Each correction step refines T* by computing an additive correction
    from backbone features, scaled by the absolute value of the physics
    increment:
        T*^{k+1} = T*^k + s_k · C_θ(h, T*^k) · |T*_init|

    The |T*_init| gate guarantees zero-preservation: when the physics
    increment T_inc_init = 0 (steady-state), the output is exactly 0
    regardless of backbone features — preventing spurious signal
    injection in near-equilibrium regions.

    Uses shared-weight correction blocks (parameter efficient) with
    per-step scale parameters.
    """
    def __init__(self, channels: int, num_iterations: int = 3):
        super().__init__()
        self.num_iterations = num_iterations
        # Shared correction network: takes h + T_estimate channel
        self.correction_net = nn.Sequential(
            nn.Conv1d(channels + 1, channels, 1),
            nn.SiLU(),
            nn.Conv1d(channels, channels // 2, 1),
            nn.SiLU(),
            nn.Conv1d(channels // 2, 1, 1),
        )
        # Per-iteration learnable step size (starts small)
        self.step_sizes = nn.Parameter(
            torch.full((num_iterations,), 0.1)
        )
        # Near-zero init on output
        nn.init.normal_(self.correction_net[-1].weight, std=1e-3)
        nn.init.zeros_(self.correction_net[-1].bias)

    def forward(self, h: torch.Tensor,
                T_inc_init: torch.Tensor,
                dt_star: float) -> torch.Tensor:
        """
        Args:
            h: backbone features [batch, channels, grid]
            T_inc_init: initial increment estimate [batch, 1, grid]
        Returns:
            T_inc_refined: [batch, 1, grid]
        """
        T_inc = T_inc_init
        # Gate: normalized spatial mask — 1 where physics has signal, 0 at steady-state
        abs_init = T_inc_init.detach().abs()
        gate_scale = abs_init.amax(dim=-1, keepdim=True).clamp(min=1e-12)
        signal_gate = abs_init / gate_scale  # [0, 1]
        for k in range(self.num_iterations):
            correction_input = torch.cat([h, T_inc], dim=1)
            delta = self.correction_net(correction_input)
            T_inc = T_inc + self.step_sizes[k] * delta * signal_gate
        return T_inc


class RelaxationGate(nn.Module):
    """Soft energy-dissipation constraint on the temperature update.

    Decomposes the increment into a free component and a relaxation
    component.  The relaxation component is soft-constrained to push
    deviation from a reference profile (linear interpolant of BCs)
    toward zero — preventing unphysical heating in cases that should
    relax.

    ΔT = ΔT_free + gate · ΔT_relax

    where ΔT_relax opposes (T_current - T_steady) and the gate is
    learned from backbone features.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Gate network: decides how much relaxation to apply per point
        self.gate_net = nn.Sequential(
            nn.Conv1d(channels + 1, channels // 2, 1),
            nn.SiLU(),
            nn.Conv1d(channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        # Relaxation strength (learned, starts moderate)
        # inverse_softplus(0.05) ≈ -2.94 so that softplus(param) ≈ 0.05
        self.relax_strength = nn.Parameter(torch.tensor(-2.94))

    def forward(self, h: torch.Tensor,
                T_inc: torch.Tensor,
                T_n_star: torch.Tensor,
                bc_left_star: torch.Tensor,
                bc_right_star: torch.Tensor,
                xi: torch.Tensor,
                dt_star: float) -> torch.Tensor:
        """
        Args:
            h: backbone features [batch, channels, grid]
            T_inc: current increment [batch, 1, grid]
            T_n_star: current temperature [batch, grid]
            bc_left_star, bc_right_star: [batch]
            xi: node positions [grid] in [0, 1]
        Returns:
            T_inc_gated: [batch, 1, grid]
        """
        # Reference: linear interpolant of BCs (steady-state for no source)
        T_ref = (bc_left_star.unsqueeze(-1)
                 + (bc_right_star - bc_left_star).unsqueeze(-1)
                 * xi.unsqueeze(0))          # [batch, grid]
        deviation = T_n_star - T_ref         # [batch, grid]

        # Relaxation direction: push deviation toward zero
        relax_dir = -deviation.unsqueeze(1)  # [batch, 1, grid]

        # Gate: how much relaxation to apply
        gate_input = torch.cat([h, T_inc], dim=1)
        gate = self.gate_net(gate_input)     # [batch, 1, grid] ∈ [0, 1]

        relax_strength = F.softplus(self.relax_strength)
        T_inc_gated = T_inc + gate * relax_strength * relax_dir
        return T_inc_gated


class SecondOrderPredictor(nn.Module):
    """Structure-preserving predictor for the Cattaneo equation.

    Predicts spatially-varying coefficients aθ(x), bθ(x) such that:
        ΔT* = aθ · sec_diff(T*_n) + bθ · (T*_n − T*_{n-1})

    Target coefficients the network should learn:
        aθ → Fo/(1+Ve)   (diffusion coefficient)
        bθ → Ve/(1+Ve)   (wave memory coefficient)

    Architecture:
        1. LNO path: spectral features of T*_n (NO residual bypass —
           spatial information must flow through the Laplace operator)
        2. GRU recurrent memory: temporal state between timesteps
        3. Coefficient MLP: predicts aθ, bθ from concatenated features
           including diff_T, Fo, Ve

    The explicit sec_diff(T*_n) and diff_T = T*_n − T*_{n-1} are
    computed from data and used structurally — the network only predicts
    the *coefficients* that multiply them, enforcing the PDE structure.
    """
    def __init__(self, channels: int, grid_size: int, modes: int = 16,
                 history_len: int = 4,
                 temporal_channels: int = 32,
                 num_no_layers: int = 4,
                 local_conv_layers: int = 2,
                 num_corrections: int = 3,
                 activation: str = 'swish',
                 spectral_filter: str = 'exponential',
                 filter_strength: float = 4.0,
                 use_spectral_norm: bool = False,
                 max_amp: float = 1.0,
                 amp_sharpness: float = 1.0,
                 pole_offset_scale: float = 0.1,
                 pole_min: float = 0.1,
                 pole_max: float = 100.0,
                 use_causal_mask: bool = True,
                 causal_safety: float = 1.0,
                 use_recurrent_memory: bool = False,
                 memory_channels: int = 32):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.modes = modes
        self.history_len = history_len
        self.temporal_channels = temporal_channels
        self.use_recurrent_memory = use_recurrent_memory
        self.memory_channels = memory_channels

        # ── LNO spatial encoder (NO residual bypass) ──
        # Projects T*_n into channel space, then processes through
        # pure LNO layers without skip connections.
        self.lno_proj = nn.Conv1d(1, channels, 1)
        _lno_kwargs = dict(
            max_amp=max_amp,
            amp_sharpness=amp_sharpness,
            pole_offset_scale=pole_offset_scale,
            pole_min=pole_min,
            pole_max=pole_max,
            use_causal_mask=use_causal_mask,
            causal_safety=causal_safety,
        )
        self.lno_blocks = nn.ModuleList()
        for _ in range(num_no_layers):
            self.lno_blocks.append(nn.ModuleDict({
                'norm': nn.InstanceNorm1d(channels),
                'spectral': LaplaceConv1d(
                    channels, channels, modes,
                    causal_mask=use_causal_mask,
                    spectral_filter=spectral_filter,
                    filter_strength=filter_strength,
                    max_amp=max_amp,
                    amp_sharpness=amp_sharpness,
                    pole_offset_scale=pole_offset_scale,
                    pole_min=pole_min,
                    pole_max=pole_max,
                    use_causal_mask=use_causal_mask,
                    causal_safety=causal_safety,
                ),
                'pointwise': nn.Conv1d(channels, channels, 1),
            }))
        self.lno_act = nn.SiLU()

        # ── GRU recurrent memory (temporal state between timesteps) ──
        # Input: projection of (T*_n, diff_T) → channels
        self.rnn_proj = nn.Conv1d(2, channels, 1)
        if use_recurrent_memory:
            self.memory_cell = Conv1dGRUCell(channels, memory_channels)
            self.memory_fusion = MemoryFusion(channels, memory_channels)

        # ── Coefficient prediction head ──
        self.coeff_head = WaveDiffusionHead(channels)

        # ── Iterative corrector (refines ΔT after structure-preserving init) ──
        self.corrector = IterativeCorrector(channels, num_iterations=num_corrections)

        # ── Relaxation gate ──
        self.relaxation_gate = RelaxationGate(channels)

    def forward(self, h_input: torch.Tensor,
                history: torch.Tensor,
                Fo_ext: torch.Tensor,
                Ve_ext: torch.Tensor,
                dt_star: float,
                dx_star: float,
                c_star: torch.Tensor,
                T_n_star: torch.Tensor,
                bc_left_star: torch.Tensor,
                bc_right_star: torch.Tensor,
                xi: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None,
                T_nm1_star_ext: Optional[torch.Tensor] = None,
                T_n_star_ext: Optional[torch.Tensor] = None,
                tau_dt_ext: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_input: projected input features [batch, channels, ext_grid]
                     (unused — kept for API compatibility; LNO has its own path)
            history: temperature history [batch, K, ext_grid] (dimensionless)
            Fo_ext, Ve_ext: [batch, ext_grid]
            dt_star, dx_star: scalars
            c_star: [batch]
            T_n_star: current temperature [batch, grid] (interior only)
            bc_left_star, bc_right_star: [batch]
            xi: node positions on interior grid [grid]
            hidden_state: recurrent memory [batch, memory_channels, ext_grid]
                or None (zero-initialised on first call)
            T_nm1_star_ext: previous temperature [batch, ext_grid]
            T_n_star_ext: current temperature [batch, ext_grid]

        Returns:
            dict with T_inc, backbone_features, coefficients, hidden_state
        """
        # ── LNO spatial path (NO residual connections) ──
        # Input: T*_n on extended grid
        x = T_n_star_ext.unsqueeze(1)  # [B, 1, ext_grid]
        x = self.lno_proj(x)            # [B, C, ext_grid]
        for block in self.lno_blocks:
            x = block['norm'](x)
            x = block['pointwise'](x)
            x = self.lno_act(x)
            x = block['spectral'](x, dt_star, dx_star, c_star)
        lno_features = x  # [B, C, ext_grid]

        # ── GRU temporal path (recurrent between timesteps) ──
        diff_T_ext = T_n_star_ext - T_nm1_star_ext  # [B, ext_grid]
        rnn_in = torch.stack([T_n_star_ext, diff_T_ext], dim=1)  # [B, 2, ext_grid]
        rnn_in = self.rnn_proj(rnn_in)  # [B, C, ext_grid]

        new_hidden = None
        h = lno_features  # Start from LNO features (no residual from input)
        if self.use_recurrent_memory:
            new_hidden = self.memory_cell(rnn_in, hidden_state)
            h = self.memory_fusion(h, new_hidden)

        # ── Coefficient prediction: aθ, bθ ──
        coeffs = self.coeff_head(h, Fo_ext, Ve_ext, tau_dt=tau_dt_ext)  # [B, 2, ext_grid]
        a_theta = coeffs[:, 0]  # [B, ext_grid] — multiplies sec_diff
        b_theta = coeffs[:, 1]  # [B, ext_grid] — multiplies diff_T

        # ── Structure-preserving update ──
        #    ΔT* = aθ · sec_diff(T*_n) + bθ · (T*_n − T*_{n-1})
        T_pad = F.pad(T_n_star_ext.unsqueeze(1), (1, 1), mode='replicate').squeeze(1)
        sec_diff_T = T_pad[:, 2:] - 2 * T_n_star_ext + T_pad[:, :-2]  # [B, ext_grid]

        T_inc_init = (a_theta * sec_diff_T + b_theta * diff_T_ext).unsqueeze(1)  # [B, 1, ext_grid]

        # ── Iterative correction ──
        T_inc = self.corrector(h, T_inc_init, dt_star)  # [B, 1, ext_grid]

        return {
            'T_inc': T_inc,                 # [B, 1, ext_grid]
            'a_theta': a_theta,             # [B, ext_grid]
            'b_theta': b_theta,             # [B, ext_grid]
            'sec_diff_T': sec_diff_T,       # [B, ext_grid]
            'backbone_features': h,         # [B, C, ext_grid]
            'hidden_state': new_hidden,     # [B, M, ext_grid] or None
        }


class CattaneoLNO(nn.Module):
    """
    Complete dimensionless Cattaneo-LNO model with ghost boundary cells
    and structural hard Dirichlet constraints.

    All inputs are converted to dimensionless form internally.
    Predictions are returned in both dimensionless and dimensional form.

    Ghost boundary cells:
        The physical domain has `grid_size` interior nodes.  Two ghost
        cells are prepended / appended to form an extended grid of size
        `grid_size + 2`.  The ghost cells carry the Dirichlet BC values,
        matching the FDM solver's ghost-node formulation in HF_Cattaneo.py.
        The LNO backbone operates on the extended grid so it naturally
        sees the boundary information.  Only the interior `grid_size`
        nodes are returned as the prediction.

    Structural hard BC constraint:
        Instead of post-hoc clamping (``T[0] = bc_left``), the predicted
        state is constructed as
            T*_{n+1} = T*_n + G_inc(ξ) + D(ξ) · N(ξ)
        where ξ ∈ [0, 1] maps boundary nodes to 0 and 1,
        G_inc is the linear interpolant of the required boundary increments,
        and D(ξ) = sin(πξ) vanishes at the boundaries.  This guarantees
        exact Dirichlet BCs for *any* network weights while preserving
        gradient flow at boundary-adjacent nodes.
    """
    def __init__(self, grid_size: int, modes: int = 16,
                 width: int = 64, num_no_layers: int = 4,
                 temporal_hidden: int = 64, num_temporal_layers: int = 2,
                 activation: str = 'swish',
                 timestep_jump: int = 1,
                 L: float = 1.0, alpha_ref: float = 1e-4,
                 T_ref: float = 200.0, delta_T: float = 100.0,
                 tau_ref: float = 1e-9,
                 spectral_filter: str = 'exponential',
                 filter_strength: float = 4.0,
                 use_ghost_cells: bool = True,
                 use_spectral_norm: bool = False,
                 max_amp: float = 1.0,
                 amp_sharpness: float = 1.0,
                 pole_offset_scale: float = 0.1,
                 pole_min: float = 0.1,
                 pole_max: float = 100.0,
                 use_causal_mask: bool = True,
                 causal_safety: float = 1.0,
                 num_internal_steps: int = 1,
                 history_len: int = 4,
                 temporal_channels: int = 32,
                 local_conv_layers: int = 2,
                 num_corrections: int = 3,
                 use_recurrent_memory: bool = False,
                 memory_channels: int = 32):
        super().__init__()
        self.grid_size = grid_size  # interior grid (physical nodes)
        self.modes = modes
        self.width = width
        self.timestep_jump = timestep_jump
        self.use_ghost_cells = use_ghost_cells
        self.num_internal_steps = max(1, num_internal_steps)
        self.activation_name = activation
        self.spectral_filter = spectral_filter
        self.filter_strength = filter_strength
        self.temporal_hidden = temporal_hidden
        self.num_temporal_layers = num_temporal_layers
        self.num_no_layers = num_no_layers
        self.history_len = history_len
        self.temporal_channels = temporal_channels
        self.use_recurrent_memory = use_recurrent_memory
        self.memory_channels = memory_channels

        # Extended grid includes 2 ghost cells when enabled
        self.extended_grid = grid_size + 2 if use_ghost_cells else grid_size

        # Dimensionless scaler for conversions
        self.scaler = DimensionlessScaler(L, alpha_ref, T_ref, delta_T, tau_ref=tau_ref)

        # Input channels (structure-preserving Fo/Ve formulation):
        #   0  T*_n           dimensionless current temperature
        #   1  T*_n − T*_{n-1} temperature difference (wave memory)
        #   2  sec_diff(T*_n) Laplacian of current temperature
        #   3  Fo             Fourier number (log1p)
        #   4  Ve             Vernotte number
        input_channels = 5

        # Input projection (kept for compatibility — the predictor
        # has its own LNO path that operates on raw T*_n)
        self.input_proj = nn.Conv1d(input_channels, width, 1)

        # Spectral-layer kwargs shared across all operator blocks
        _lno_kwargs = dict(
            max_amp=max_amp,
            amp_sharpness=amp_sharpness,
            pole_offset_scale=pole_offset_scale,
            pole_min=pole_min,
            pole_max=pole_max,
            use_causal_mask=use_causal_mask,
            causal_safety=causal_safety,
        )

        # Structure-preserving predictor with LNO spatial encoder (no
        # residual bypass), GRU recurrent memory, and coefficient MLP.
        self.predictor = SecondOrderPredictor(
            width, self.extended_grid, modes,
            history_len=history_len,
            temporal_channels=temporal_channels,
            num_no_layers=num_no_layers,
            local_conv_layers=local_conv_layers,
            num_corrections=num_corrections,
            activation=activation,
            spectral_filter=spectral_filter,
            filter_strength=filter_strength,
            use_spectral_norm=use_spectral_norm,
            use_recurrent_memory=use_recurrent_memory,
            memory_channels=memory_channels,
            **_lno_kwargs,
        )

        # Stability constraint (buffer, not Parameter — non-differentiable
        # and must not drift under weight decay)
        self.register_buffer('cfl_threshold', torch.tensor(0.5))

        # Cache for the extended spatial coordinate tensor.  Rebuilt only
        # when the grid size, dx_star, or device changes — not every forward.
        self._cached_x_coord_key: Optional[tuple] = None
        self._cached_x_coord: Optional[torch.Tensor] = None

        # Cache for the structural BC constraint coordinate / distance
        # tensors.  xi ∈ [0, 1] with xi[0]=0 and xi[-1]=1  (not cell-centred)
        # so that D(xi) = xi*(1-xi) vanishes exactly at the boundary nodes.
        self._cached_bc_key: Optional[tuple] = None
        self._cached_bc_xi: Optional[torch.Tensor] = None   # [grid]
        self._cached_bc_D: Optional[torch.Tensor] = None     # [grid]

    def compute_dimensionless_wave_speed(self, Fo: float, Ve: float) -> torch.Tensor:
        """
        Compute dimensionless wave speed from Fo and Ve.

        In the Fo/Ve formulation:
            c* ≈ Ve / sqrt(Fo)   (approximate, for CFL check)

        Args:
            Fo: Fourier number (scalar)
            Ve: Vernotte number (scalar)

        Returns:
            c_star: Dimensionless wave speed (scalar tensor)
        """
        c_star = Ve / math.sqrt(Fo + 1e-15)
        return c_star

    def check_stability(self, dt_star: float, dx_star: float, 
                       c_star: torch.Tensor) -> torch.Tensor:
        """
        Check CFL condition in dimensionless form.

        CFL = c* · dt* / dx* < threshold
        """
        cfl = c_star * dt_star / dx_star
        stable = cfl < self.cfl_threshold
        return stable

    def forward(self, T_n: torch.Tensor, T_nm1: torch.Tensor,
                q: torch.Tensor, tau: torch.Tensor, alpha: torch.Tensor,
                rho_cp: torch.Tensor, bc_left: torch.Tensor, bc_right: torch.Tensor,
                dt: float, dx: float,
                T_history: Optional[torch.Tensor] = None,
                hidden_state: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional multi-step time integration.

        When ``num_internal_steps`` > 1 the macro timestep *dt* is split
        into N equal substeps.  The same backbone is applied at each
        substep with a proportionally reduced Fo, and each substep's
        output feeds into the next.  This converts the single amortised
        map into a learned time integrator that compounds well over long
        autoregressive rollouts.

        Args:
            hidden_state: recurrent memory from the previous autoregressive
                step, shape ``[B, memory_channels, ext_grid]``.  Pass *None*
                for the first step or when recurrent memory is disabled.
        """
        if self.num_internal_steps <= 1:
            return self._single_step_forward(
                T_n, T_nm1, q, tau, alpha, rho_cp,
                bc_left, bc_right, dt, dx, T_history,
                hidden_state=hidden_state)

        # ── Multi-step time integration ──
        dt_sub = dt / self.num_internal_steps
        T_curr = T_n
        T_prev = T_nm1
        h_state = hidden_state

        for _ in range(self.num_internal_steps):
            result = self._single_step_forward(
                T_curr, T_prev, q, tau, alpha, rho_cp,
                bc_left, bc_right, dt_sub, dx,
                hidden_state=h_state)
            T_prev = T_curr
            T_curr = result['T_pred']
            h_state = result.get('hidden_state')

        # Total increment relative to original T_n (for data / physics loss)
        T_n_star = self.scaler.to_dimensionless_temp(T_n)
        T_curr_star = self.scaler.to_dimensionless_temp(T_curr)
        total_inc_star = T_curr_star - T_n_star

        # Return macro-level Fo so the physics loss uses the Fourier
        # number that matches the training data's full dt.
        Fo_macro = DimensionlessScaler.compute_Fo_field(
            alpha, dt, dx,
            Fo_min=self.scaler.Fo_min,
            Fo_max=self.scaler.Fo_max)

        result['T_pred'] = T_curr
        result['T_pred_star'] = total_inc_star
        result['T_n_star'] = T_n_star
        result['T_increment'] = total_inc_star * self.scaler.delta_T
        result['Fo'] = Fo_macro
        # Substep Fo for coefficient regularization (a_theta targets Fo_sub/(1+Ve))
        result['Fo_substep'] = Fo_macro / self.num_internal_steps

        return result

    def _single_step_forward(
                self, T_n: torch.Tensor, T_nm1: torch.Tensor,
                q: torch.Tensor, tau: torch.Tensor, alpha: torch.Tensor,
                rho_cp: torch.Tensor, bc_left: torch.Tensor, bc_right: torch.Tensor,
                dt: float, dx: float,
                T_history: Optional[torch.Tensor] = None,
                hidden_state: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """Single time-integration substep with second-order velocity update.

        Architecture flow:
        1. Build history window (mandatory — padded from T_n/T_nm1 if needed)
        2. Input projection → hidden features
        3. SecondOrderPredictor: temporal conv → multi-scale LNO blocks →
        Architecture flow (structure-preserving):
        1. Convert to dimensionless form, compute Fo/Ve fields
        2. Build ghost cells and extended grid
        3. LNO spatial encoder + GRU memory → predict aθ, bθ
        4. Structure-preserving update: ΔT = aθ·∇²T + bθ·diff_T
        5. Iterative correction + relaxation gate
        6. Hard Dirichlet BC constraint
        """
        batch_size = T_n.shape[0]
        actual_grid = T_n.shape[1]  # may differ from self.grid_size at inference

        # Convert to dimensionless form
        T_n_star = self.scaler.to_dimensionless_temp(T_n)
        T_nm1_star = self.scaler.to_dimensionless_temp(T_nm1)
        bc_left_star = self.scaler.to_dimensionless_temp(bc_left)
        bc_right_star = self.scaler.to_dimensionless_temp(bc_right)
        net_dtype = self.input_proj.weight.dtype

        # ── Spatially varying Fo / Ve / τ/dt computation ──
        Fo_field = DimensionlessScaler.compute_Fo_field(
            alpha, dt, dx,
            Fo_min=self.scaler.Fo_min,
            Fo_max=self.scaler.Fo_max)          # [batch, grid]
        Ve_field = DimensionlessScaler.compute_Ve_field(
            alpha, tau, self.scaler.L)          # [batch, grid]
        tau_dt_field = tau / dt                  # [batch, grid] — discrete coupling

        dx_star = self.scaler.get_dx_star(dx)
        dt_star = self.scaler.get_dt_star(dt)

        T_n_star_net = T_n_star.to(dtype=net_dtype)
        T_nm1_star_net = T_nm1_star.to(dtype=net_dtype)
        bc_left_star_net = bc_left_star.to(dtype=net_dtype)
        bc_right_star_net = bc_right_star.to(dtype=net_dtype)

        # ── Ghost boundary cell construction (fused) ──
        Fo_net = Fo_field.to(dtype=net_dtype)
        Ve_net = Ve_field.to(dtype=net_dtype)
        tau_dt_net = tau_dt_field.to(dtype=net_dtype)
        if self.use_ghost_cells:
            # Batch T_n and T_nm1 into single pad → slice apart
            T_both = torch.stack([T_n_star_net, T_nm1_star_net], dim=0)  # [2, B, G]
            T_both_ext = F.pad(T_both, (1, 1))  # [2, B, G+2]
            T_both_ext[:, :, 0] = bc_left_star_net
            T_both_ext[:, :, -1] = bc_right_star_net
            T_n_ext = T_both_ext[0]    # [B, G+2]
            T_nm1_ext = T_both_ext[1]  # [B, G+2]
            # Replicate-pad Fo/Ve/tau_dt by copying edge values
            Fo_ext = F.pad(Fo_net, (1, 1), mode='replicate')
            Ve_ext = F.pad(Ve_net, (1, 1), mode='replicate')
            tau_dt_ext = F.pad(tau_dt_net, (1, 1), mode='replicate')
            ext_grid = actual_grid + 2
        else:
            T_n_ext = T_n_star_net
            T_nm1_ext = T_nm1_star_net
            Fo_ext = Fo_net
            Ve_ext = Ve_net
            tau_dt_ext = tau_dt_net
            ext_grid = actual_grid

        # ── Per-sample dimensionless wave speed (data-dependent) ──
        L = self.scaler.L
        alpha_ref = self.scaler.alpha_ref
        c_star_field = torch.sqrt(alpha / (tau + 1e-30)) * L / alpha_ref
        c_star = c_star_field.max(dim=-1).values  # [batch]

        dt_star_macro = dt_star

        # ── BC constraint coordinates (cached) ──
        _bc_key = (actual_grid, T_n.device.type, getattr(T_n.device, 'index', None))
        if self._cached_bc_key != _bc_key:
            xi = torch.linspace(0.0, 1.0, actual_grid, device=T_n.device)
            self._cached_bc_xi = xi
            # Step mask: 0 at boundary nodes, 1 at all interior nodes.
            # sin(πξ) attenuated wave-memory term near boundaries; step avoids this.
            D = torch.ones(actual_grid, device=T_n.device)
            D[0] = 0.0
            D[-1] = 0.0
            self._cached_bc_D = D
            self._cached_bc_key = _bc_key
        xi = self._cached_bc_xi
        D = self._cached_bc_D

        # ── Structure-preserving prediction ──
        pred_out = self.predictor(
            None, None, Fo_ext, Ve_ext,
            dt_star_macro, dx_star, c_star,
            T_n_star, bc_left_star, bc_right_star, xi,
            hidden_state=hidden_state,
            T_nm1_star_ext=T_nm1_ext,
            T_n_star_ext=T_n_ext,
            tau_dt_ext=tau_dt_ext,
        )

        T_inc_ext = pred_out['T_inc']        # [B, 1, ext_grid]
        backbone_h = pred_out['backbone_features']  # [B, channels, ext_grid]
        new_hidden = pred_out['hidden_state']  # [B, M, ext_grid] or None

        # ── Slice back to interior nodes ──
        if self.use_ghost_cells:
            T_inc_interior = T_inc_ext.squeeze(1)[:, 1:-1]
            backbone_interior = backbone_h[:, :, 1:-1]
        else:
            T_inc_interior = T_inc_ext.squeeze(1)
            backbone_interior = backbone_h

        # ── Relaxation gate (on interior grid) ──
        T_inc_interior_gated = self.predictor.relaxation_gate(
            backbone_interior, T_inc_interior.unsqueeze(1),
            T_n_star, bc_left_star, bc_right_star, xi, dt_star_macro,
        ).squeeze(1)  # [B, grid]

        # ── Structural hard Dirichlet constraint ──
        #   T*_{n+1} = T*_n + G_inc(ξ) + D(ξ) · N(ξ)
        bc_inc_left = bc_left_star - T_n_star[:, 0]
        bc_inc_right = bc_right_star - T_n_star[:, -1]

        G_inc = (bc_inc_left.unsqueeze(-1)
                 + (bc_inc_right - bc_inc_left).unsqueeze(-1) * xi.unsqueeze(0))

        shaped_increment = G_inc + D.unsqueeze(0) * T_inc_interior_gated

        T_full_star = T_n_star + shaped_increment

        # Convert back to dimensional form
        T_pred = self.scaler.from_dimensionless_temp(T_full_star)

        T_pred_star_out = shaped_increment

        # Check stability
        stable = self.check_stability(dt_star_macro, dx_star, c_star)
        t_c = self.scaler.t_c
        c_dimensional = c_star * self.scaler.L / t_c

        return {
            'T_pred': T_pred,
            'wave_speed': c_dimensional,
            'stable': stable,
            'T_increment': T_pred_star_out * self.scaler.delta_T,
            'T_pred_star': T_pred_star_out,
            'T_n_star': T_n_star,
            'Fo': Fo_field,
            'Ve': Ve_field,
            'dx_star': dx_star,
            'hidden_state': new_hidden,
            'a_theta': pred_out.get('a_theta'),
            'b_theta': pred_out.get('b_theta'),
            'tau_dt': tau_dt_field,
        }

class AdaptiveLossWeights(nn.Module):
    """
    Adaptive loss weighting for combining data and physics losses dynamically.
    Helps balance physics constraints with actual PDE sequence matching.
    """
    def __init__(self, num_losses: int = 5, ema_decay: float = 0.9,
                 temperature: float = 1.0, strategy: str = 'magnitude',
                 warmup_epochs: int = 10,
                 **kwargs):
        super().__init__()
        self.num_losses = num_losses
        self.ema_decay = ema_decay
        self.temperature = temperature
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs

        # Loss names mapping
        # 0: data, 1: cattaneo, 2: energy, 3: characteristic, 4: bc
        self.register_buffer('ema_losses', torch.ones(num_losses))
        self.register_buffer('weights', torch.ones(num_losses))

        # In uncertainty strategy, we learn the log variances
        if self.strategy == 'uncertainty':
            self.log_vars = nn.Parameter(torch.zeros(num_losses))
            
    def update(self, current_losses: torch.Tensor):
        
        if self.strategy == 'magnitude':
            # Update EMA of losses
            self.ema_losses = self.ema_decay * self.ema_losses + (1 - self.ema_decay) * current_losses.detach()
            
            # Compute inverse magnitude weights with temperature
            eps = 1e-8
            avg_loss = self.ema_losses.mean()
            raw_weights = (avg_loss / (self.ema_losses + eps)) ** self.temperature
            # raw_weights = ((self.ema_losses + eps) / (avg_loss + eps)) ** (self.temperature)
            
            # Clamp and normalize
            raw_weights = torch.clamp(raw_weights, 0.001, 100.0)
            self.weights = raw_weights / raw_weights.sum() * self.num_losses
            
        elif self.strategy == 'uncertainty':
            # Weights are derived from learned log variances: w_i = exp(-log_var_i)
            with torch.no_grad():
                self.weights = torch.exp(-self.log_vars)

    def get_weights(self) -> torch.Tensor:
        return self.weights

class CattaneoPhysicsLoss(nn.Module):
    """
    Physics-informed loss functions for the Cattaneo equation using
    spatially varying Fourier number Fo(x) and relaxation time τ(x).

    Each training sample has spatially varying material properties
    α(x) and τ(x), so Fo is computed as a **field** (same shape
    as the temperature grid) from the actual alpha stored in the
    targets dict.  This ensures the physics residual coefficients
    match the FDM solver that generated the training data at every
    grid point.

    Dimensional Cattaneo PDE (constant-property, discrete):
        τ·(T−2Tₙ+Tₙ₋₁)/dt² + (T−Tₙ)/dt = α·(T_{j+1}−2T_j+T_{j-1})/dx²

    Nondimensionalising T* = (T−T_ref)/ΔT and dividing by ΔT/dt gives
    the discrete residual at grid point j of sample i:

        R_{i,j} = (τ/dt)·(T* − 2T*_prev + T*_prev2)
                + (T* − T*_prev)
                − Fo · sec_diff(T*)
                − q̂

    where sec_diff(T*) = T*_{j+1} − 2T*_j + T*_{j-1}  (NOT divided by dx*²)
          Fo = α·dt/dx²
          q̂  = q·dt / (ρcₚ·ΔT)   dimensionless source per timestep
    """
    def __init__(self, lambda_cattaneo: float = 0.1,
                 lambda_energy: float = 0.1,
                 lambda_characteristic: float = 0.1,
                 lambda_bc: float = 1.0,
                 lambda_dTdt: float = 0.0,
                 data_loss_floor_k: float = 1e-3,
                 lambda_gain: float = 1.0,
                 L: float = 1.0, alpha_ref: float = 1e-4,
                 T_ref: float = 200.0, delta_T: float = 100.0,
                 tau_ref: float = 1e-9):
        super().__init__()
        self.lambda_cattaneo = lambda_cattaneo
        self.lambda_energy = lambda_energy
        self.lambda_characteristic = lambda_characteristic
        self.lambda_bc = lambda_bc
        self.lambda_dTdt = lambda_dTdt
        self.data_loss_floor_k = data_loss_floor_k
        self.lambda_gain = lambda_gain

        # Physics warmup: multiplicative factor applied to ALL physics
        # loss components.  Ramped from 0 → 1 during early training so
        # that the model first learns from data before physics
        # constraints are enforced.  Set by the trainer via
        # set_physics_warmup().
        self.physics_warmup_factor = 1.0

        # Dimensionless scaler for temperature / q conversions
        self.scaler = DimensionlessScaler(L, alpha_ref, T_ref, delta_T, tau_ref=tau_ref)
        self.data_loss_floor_star_sq = (data_loss_floor_k / self.scaler.delta_T) ** 2

    def set_physics_warmup(self, factor: float):
        """Set the physics warmup factor (0 = no physics, 1 = full physics)."""
        self.physics_warmup_factor = max(0.0, min(1.0, factor))

    @staticmethod
    def _compute_sec_diff(T_star: torch.Tensor) -> torch.Tensor:
        """Spatial second difference with replicate boundary padding."""
        T_pad = F.pad(T_star.unsqueeze(1), (1, 1), mode='replicate').squeeze(1)
        return T_pad[:, 2:] - 2 * T_star + T_pad[:, :-2]

    def cattaneo_residual_star(self, inc_star: torch.Tensor,
                               inc_prev_star: torch.Tensor,
                               T_full_star: torch.Tensor,
                               q_star: torch.Tensor,
                               Fo: torch.Tensor, tau_dt: torch.Tensor,
                               sec_diff_T: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Discrete Cattaneo residual.

        R = (τ/dt)·(ΔTⁿ − ΔTⁿ⁻¹) + ΔTⁿ − Fo·∇²Tⁿ⁺¹ − q*

        Args:
            inc_star:      ΔTⁿ = T*_{n+1} − T*_n        [batch, grid]
            inc_prev_star: ΔTⁿ⁻¹ = T*_n − T*_{n-1}      [batch, grid]
            T_full_star:   T*_{n+1}                       [batch, grid]
            q_star:        dimensionless source            [batch, grid]
            Fo:            Fourier number α·dt/dx²         [batch, grid]
            tau_dt:        discrete coupling τ/dt           [batch, grid]

        Returns:
            residual: [batch, grid]
        """
        if sec_diff_T is None:
            sec_diff_T = self._compute_sec_diff(T_full_star)

        residual = (tau_dt * (inc_star - inc_prev_star)
                   + inc_star
                   - Fo * sec_diff_T
                   - q_star)
        return residual


    def energy_conservation_star(self, inc_star: torch.Tensor,
                                  T_full_star: torch.Tensor,
                                  q_star: torch.Tensor,
                                  Fo: torch.Tensor, dt: float,
                                  sec_diff_T: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Energy conservation in increment form.

        <inc − Fo·∇²T*_{n+1} − q*> ≈ 0

        All terms O(inc) ~ O(1e-4), consistent with the residual scale.

        Args:
            inc_star: predicted increment  [batch, grid]
            T_full_star: T*_n + inc_star  [batch, grid]
            Fo: [batch, grid] spatially varying
            dt: time step [s]
        """
        if sec_diff_T is None:
            sec_diff_T = self._compute_sec_diff(T_full_star)
        diffusion = Fo * sec_diff_T                                  # [batch, grid]
        energy_density = inc_star - diffusion - q_star               # [batch, grid]
        conservation_star = energy_density.mean(dim=-1)             # [batch]
        return conservation_star

    def characteristic_residual_star(self, T_star: torch.Tensor, T_prev_star: torch.Tensor,
                                    Fo: torch.Tensor, tau: torch.Tensor,
                                    dt: float, dx_star: float) -> torch.Tensor:
        """Characteristic residual along the Cattaneo wave characteristics.

        The Cattaneo equation has characteristic speed c = √(α/τ).
        In dimensionless form, the characteristic condition is:
            dT* ± cfl · dT*/dx* ≈ 0   along characteristics

        where cfl = √(Fo·dt/τ).

        We normalise by (1 + cfl/dx_star) so the two terms are O(1).

        Args:
            Fo: [batch, grid] spatially varying
            tau: [batch, grid] relaxation time [s]
            dt: time step [s]
            dx_star: scalar (dx/L)
        """
        cfl = torch.sqrt(Fo * dt / (tau + 1e-30))  # [batch, grid]

        dT = T_star - T_prev_star              # [batch, grid]
        dTdx_star = (T_star[:, 1:] - T_star[:, :-1]) / dx_star
        dTdx_star = F.pad(dTdx_star, (0, 1), mode='replicate')

        raw_characteristic = dT + cfl * dTdx_star

        return raw_characteristic

    
    def boundary_residual_star(self, T_star: torch.Tensor,
                               bc_left_star: torch.Tensor,
                               bc_right_star: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Boundary conditions in dimensionless form."""
        left_residual = T_star[:, 0] - bc_left_star
        right_residual = T_star[:, -1] - bc_right_star
        return left_residual, right_residual

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                adaptive_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute total physics-informed loss using spatially varying Fo/Ve fields."""

        # ── Ensure fp32 for loss computation ──
        # Under AMP autocast, model outputs may be in reduced precision.
        # Squared-error terms can overflow fp16 (max 65504) with untrained
        # predictions, producing NaN.  Casting to fp32 is a no-op when
        # predictions are already fp32.
        predictions = {
            k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in predictions.items()
        }

        # Extract dimensionless temperature fields (always fp32)
        T_true_star = self.scaler.to_dimensionless_temp(targets['T']).float()
        T_prev_star = self.scaler.to_dimensionless_temp(targets['T_prev']).float()
        T_prev2_star = self.scaler.to_dimensionless_temp(targets['T_prev2']).float()
        bc_left_star = self.scaler.to_dimensionless_temp(targets['bc_left']).float()
        bc_right_star = self.scaler.to_dimensionless_temp(targets['bc_right']).float()

        # Dimensionless heat source: q* = q / q_c
        q_star = self.scaler.to_dimensionless_q(targets['q']).float()

        dt = targets['dt']
        dx = targets['dx']

        # ── SPATIALLY VARYING Fo FIELD ──
        alpha = targets['alpha'].float()   # [batch, grid]
        tau = targets['tau'].float()       # [batch, grid]

        Fo = predictions.get('Fo')
        if Fo is None:
            Fo = DimensionlessScaler.compute_Fo_field(
                alpha, dt, dx,
                Fo_min=self.scaler.Fo_min,
                Fo_max=self.scaler.Fo_max)  # [batch, grid]
        dx_star = self.scaler.get_dx_star(dx)

        # Get predicted temperature (handle both increment and full forms)
        if 'T_pred_star' in predictions and 'T_n_star' in predictions:
            T_pred_star_inc = predictions['T_pred_star']
            T_n_star = predictions['T_n_star']
            T_full_star = T_pred_star_inc + T_n_star
        elif 'T_pred_star' in predictions:
            T_full_star = predictions['T_pred_star']
            T_pred_star_inc = None
        else:
            T_full_star = self.scaler.to_dimensionless_temp(predictions['T_pred'])
            T_pred_star_inc = None

        # ── Data loss: absolute MSE on increments + gain penalty ──
        T_true_inc_star = T_true_star - T_prev_star
        if T_pred_star_inc is not None:
            T_pred_inc_star = T_pred_star_inc
            inc_error_sq = (T_pred_inc_star - T_true_inc_star).pow(2).mean(dim=-1)
        else:
            T_pred_inc_star = T_full_star - T_prev_star
            inc_error_sq = (T_full_star - T_true_star).pow(2).mean(dim=-1)
        loss_shape = inc_error_sq.mean()
        # Gain penalty: penalise deviation from unit gain ratio
        inc_target_sq_mean = T_true_inc_star.detach().pow(2).mean(dim=-1)
        inc_scale_sq = inc_target_sq_mean.clamp(min=self.data_loss_floor_star_sq)
        inc_pred_energy = T_pred_inc_star.pow(2).mean(dim=-1)
        gain_ratio = (inc_pred_energy / inc_scale_sq.detach()).clamp(min=1e-12).sqrt()
        loss_gain = (gain_ratio - 1.0).pow(2).mean()
        loss_data = loss_shape + self.lambda_gain * loss_gain

        # ── Physics warmup factor (used for curriculum) ──
        pwf = self.physics_warmup_factor  # 0 → 1 during early training

        # ── Physics losses ──
        _zero = torch.tensor(0.0, device=loss_data.device)
        # Precompute spatial second difference once for reuse
        sec_diff_T = self._compute_sec_diff(T_full_star)

        # ── τ/dt field (discrete coupling factor for Cattaneo) ──
        tau_dt = predictions.get('tau_dt')
        if tau_dt is None:
            tau_dt = tau / dt

        # ── Ve field (Vernotte number — only for fallback) ──
        Ve = predictions.get('Ve')
        if Ve is None:
            Ve = DimensionlessScaler.compute_Ve_field(
                alpha, tau, self.scaler.L)

        # ── Increments for 1st-order residual ──
        inc_star = T_full_star - T_prev_star       # predicted increment ΔTⁿ
        inc_prev_star = T_prev_star - T_prev2_star  # previous ΔTⁿ⁻¹ (from data)

        # ── Cattaneo residual: R = (τ/dt)·(ΔTⁿ − ΔTⁿ⁻¹) + ΔTⁿ − Fo·∇²T − q* ──
        residual_star = self.cattaneo_residual_star(
            inc_star, inc_prev_star, T_full_star,
            q_star, Fo, tau_dt,
            sec_diff_T=sec_diff_T
        )
        loss_cattaneo = torch.nan_to_num((residual_star ** 2).mean(), nan=0.0)

        # ── Energy conservation (increment form) ──
        conservation_star = self.energy_conservation_star(
            inc_star, T_full_star, q_star, Fo, dt,
            sec_diff_T=sec_diff_T
        )
        loss_energy = torch.nan_to_num((conservation_star ** 2).mean(), nan=0.0)

        # ── Characteristic residual ──
        characteristic_star = self.characteristic_residual_star(
            T_full_star, T_prev_star, Fo, tau, dt, dx_star
        )
        loss_characteristic = torch.nan_to_num((characteristic_star ** 2).mean(), nan=0.0)

        # ── BC loss ──
        if self.lambda_bc > 0:
            left_res, right_res = self.boundary_residual_star(
                T_full_star, bc_left_star, bc_right_star
            )
            loss_bc = (left_res ** 2).mean() + (right_res ** 2).mean()
        else:
            loss_bc = _zero

        # ── Coefficient regularization: penalise a_theta deviation from physics ──
        # b_theta is frozen at physics target, so only a needs regularization.
        loss_coeff_reg = _zero
        if 'a_theta' in predictions:
            a_theta = predictions['a_theta'].float()
            # Use τ/dt for physics targets (matches coefficient head)
            denom = (1.0 + tau_dt).clamp(min=1e-6)
            Fo_coeff = predictions.get('Fo_substep', Fo)
            a_target = Fo_coeff / denom
            # Interior grid only (trim ghost cells if present)
            G = T_full_star.shape[-1]
            a_t = a_target[:, :G]
            a_p = a_theta[:, :G]
            loss_coeff_reg = (
                ((a_p - a_t) / a_t.abs().clamp(min=1e-6)).pow(2).mean()
            )

        # ── Combined physics loss (raw — gradient balancing handles scale) ──
        loss_physics = (self.lambda_cattaneo * loss_cattaneo +
                        self.lambda_energy * loss_energy +
                        self.lambda_characteristic * loss_characteristic)

        # ── Loss assembly: gradient-balanced weighting done in trainer ──
        # Return separate data/physics for the trainer to balance.
        # Default assembly (when trainer doesn't do gradient balancing):
        loss_total = loss_data + pwf * loss_physics + self.lambda_bc * loss_bc

        return {
            'loss_total': loss_total,
            'loss_data': loss_data,
            'loss_cattaneo': loss_cattaneo,
            'loss_energy': loss_energy,
            'loss_characteristic': loss_characteristic,
            'loss_bc': loss_bc,
            'loss_physics': loss_physics,
            'loss_coeff_reg': loss_coeff_reg,
            'Fo': Fo,
        }

def create_cattaneo_model(grid_size: int, L: float = 1.0, 
                          alpha_ref: float = 1e-4,
                          T_ref: float = 200.0, delta_T: float = 100.0,
                          tau_ref: float = 1e-9,
                          **kwargs) -> CattaneoLNO:
    """
    Factory function to create dimensionless Cattaneo-LNO model.

    Args:
        grid_size: Number of grid points
        L: Characteristic length (domain size) [m]
        alpha_ref: Reference thermal diffusivity [m²/s]
        T_ref: Reference temperature [K]
        delta_T: Characteristic temperature difference [K]
        tau_ref: Reference relaxation time [s] (sets balanced timescale)
        **kwargs: Additional config overrides including:
            spectral_filter: 'exponential', 'raised_cosine', 'learnable',
                             'sharp_cutoff', or 'none'
            filter_strength: Strength parameter for spectral filter
            use_ghost_cells: Whether to use ghost boundary cells
    """
    default_config = {
        'modes': 16,
        'width': 64,
        'num_no_layers': 4,
        'temporal_hidden': 64,
        'num_temporal_layers': 2,
        'activation': 'swish',
        'timestep_jump': 1,
        'L': L,
        'alpha_ref': alpha_ref,
        'T_ref': T_ref,
        'delta_T': delta_T,
        'tau_ref': tau_ref,
        'spectral_filter': 'exponential',
        'filter_strength': 4.0,
        'use_ghost_cells': True,
        'use_spectral_norm': False,
        # Polar weight / data-dependent pole / causal mask defaults
        'max_amp': 1.0,
        'amp_sharpness': 1.0,
        'pole_offset_scale': 0.1,
        'pole_min': 0.1,
        'pole_max': 100.0,
        'use_causal_mask': True,
        'causal_safety': 1.0,
        'num_internal_steps': 1,
        # Second-order architecture defaults
        'history_len': 4,
        'temporal_channels': 32,
        'local_conv_layers': 2,
        'num_corrections': 3,
        # Recurrent memory defaults (off by default for backward compat)
        'use_recurrent_memory': False,
        'memory_channels': 32,
    }
    default_config.update(kwargs)

    return CattaneoLNO(grid_size, **default_config)


if __name__ == '__main__':
    # Test the dimensionless model with ghost boundary cells
    batch_size = 4
    grid_size = 112

    # Physical scales
    L = 1.12e-6  # 112 * 1e-8
    alpha_ref = 1e-4
    T_ref = 200.0
    delta_T = 100.0

    model = create_cattaneo_model(grid_size, L, alpha_ref, T_ref, delta_T,
                                   spectral_filter='exponential',
                                   filter_strength=4.0,
                                   use_ghost_cells=True)

    # Create dimensional inputs (on the physical grid, not extended)
    T_n = torch.randn(batch_size, grid_size) * 50 + 200
    T_nm1 = torch.randn(batch_size, grid_size) * 50 + 200
    q = torch.randn(batch_size, grid_size) * 1e8
    tau = torch.ones(batch_size, grid_size) * 1e-9
    alpha = torch.ones(batch_size, grid_size) * 1e-4
    rho_cp = torch.ones(batch_size, grid_size) * 1e6
    bc_left = torch.ones(batch_size) * 100.0
    bc_right = torch.ones(batch_size) * 200.0

    # Dimensional steps
    dt = 1e-13
    dx = L / grid_size

    # Forward pass
    output = model(T_n, T_nm1, q, tau, alpha, rho_cp, bc_left, bc_right, dt, dx)

    print("Model output shapes:")
    for key, val in output.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}, range: [{val.min():.3e}, {val.max():.3e}]")
        else:
            print(f"  {key}: {val}")

    # Verify output grid matches input grid (ghost cells are internal)
    assert output['T_pred'].shape == (batch_size, grid_size), \
        f"Expected ({batch_size}, {grid_size}), got {output['T_pred'].shape}"
    print(f"\n✓ Output grid matches physical grid ({grid_size} nodes)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Extended grid (internal): {model.extended_grid}")
    print(f"LNO spectral filter: {model.spectral_filter}")

    # Check dimensionless values (Fo/Ve formulation)
    Fo = model.scaler.compute_Fo(alpha_ref, dt, dx)
    Ve = model.scaler.compute_Ve(alpha_ref, 1e-9, L)
    tau_over_dt = 1e-9 / dt
    print(f"\nDimensionless check:")
    print(f"  Fo = α·dt/dx² = {Fo:.3e}")
    print(f"  Ve = √(α·τ)/L = {Ve:.3e}")
    print(f"  τ/dt     = {tau_over_dt:.3e}  (inertia coefficient)")
    print(f"  Fo       = {Fo:.3e}  (diffusion coefficient on sec_diff)")
    print(f"  dx* = dx/L = {model.scaler.get_dx_star(dx):.3e}")
    print(f"  Input channels: 5 (structure-preserving Fo/Ve formulation)")
