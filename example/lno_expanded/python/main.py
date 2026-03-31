"""
Main Workflow for Cattaneo-LNO
===============================

Three-step workflow
-------------------
  1. generate  - Generate training data for the Cattaneo system
  2. train     - Train the LNO model (with optional WandB logging)
  3. benchmark - Run the trained model vs pure FDM in every inference mode

Convenience shortcut
--------------------
  all - Run steps 1, 2, 3 sequentially

Individual inference modes (require a trained checkpoint)
---------------------------------------------------------
  hybrid     - Hybrid solver (neural + FDM fallback) vs pure FDM
  super-res  - Super-resolution inference vs fine-grid FDM
  frame-gen  - Temporal frame generation vs FDM
  warm-start - Neural warm-start + FDM refinement vs cold-start FDM

WandB integration
-----------------
  Pass --wandb to enable Weights & Biases logging during training.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from cattaneo_lno import create_cattaneo_model
from data_generation import generate_training_data
from evaluate import load_model
from train_cattaneo_lno import CattaneoTrainer

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_alpha_ref(T_ref: float = 200.0, material_file: int = 1) -> float:
    """Derive alpha_ref from the material data file at T_ref."""
    from HF_Cattaneo import _load_material
    mat = _load_material(material_file)
    T_grid = mat['T_grid']
    slope = (len(T_grid) - 1) / (mat['T_max'] - mat['T_min'])
    idx = int((T_ref - mat['T_min']) * slope)
    idx = max(0, min(len(T_grid) - 2, idx))
    t = (T_ref - T_grid[idx]) / (T_grid[idx + 1] - T_grid[idx])
    Cv = float(mat['hc_values'][idx] * (1 - t) + mat['hc_values'][idx + 1] * t)
    k = float(mat['tk_values'][idx] * (1 - t) + mat['tk_values'][idx + 1] * t)
    return k / Cv


def setup_directories():
    for d in ['checkpoints', 'data', 'results', 'figures']:
        Path(d).mkdir(exist_ok=True)


def _build_model(config: dict):
    """Construct a CattaneoLNO from the config dict, optionally wrapped
    with an autoencoder for large grids."""
    L = config['grid_size'] * config['dx']

    # If autoencoder is enabled, the LNO operates on a smaller latent grid
    use_ae = config.get('use_autoencoder', False)
    if use_ae:
        from autoencoder import wrap_model_with_autoencoder, compute_compression_stats

        latent_grid = config.get('ae_latent_grid', config['grid_size'] // 4)
        lno_grid = latent_grid
        lno_L = lno_grid * config['dx'] * (config['grid_size'] / lno_grid)
    else:
        lno_grid = config['grid_size']
        lno_L = L

    lno = create_cattaneo_model(
        grid_size=lno_grid,
        modes=config['modes'],
        width=config['width'],
        num_no_layers=config.get('num_no_layers', 2),
        temporal_hidden=config.get('temporal_hidden', 16),
        num_temporal_layers=config.get('num_temporal_layers', 2),
        activation=config.get('activation', 'swish'),
        timestep_jump=config['timestep_jump'],
        L=lno_L,
        alpha_ref=config.get('alpha_ref', _get_alpha_ref()),
        T_ref=200.0,
        delta_T=100.0,
        tau_ref=config.get('tau', 1e-9),
        spectral_filter=config.get('spectral_filter', 'None'),
        filter_strength=config.get('filter_strength', 1.0),
        use_ghost_cells=config.get('use_ghost_cells', True),
        use_spectral_norm=config.get('use_spectral_norm', False),
        max_amp=config.get('max_amp', 1.0),
        amp_sharpness=config.get('amp_sharpness', 1.0),
        pole_offset_scale=config.get('pole_offset_scale', 0.1),
        pole_min=config.get('pole_min', 0.1),
        pole_max=config.get('pole_max', 100.0),
        use_causal_mask=config.get('use_causal_mask', False),
        causal_safety=config.get('causal_safety', 1.0),
        num_internal_steps=config.get('num_internal_steps', 1),
        history_len=config.get('history_len', 4),
        temporal_channels=config.get('temporal_channels', 32),
        local_conv_layers=config.get('local_conv_layers', 2),
        num_corrections=config.get('num_corrections', 3),
        use_recurrent_memory=config.get('use_recurrent_memory', False),
        memory_channels=config.get('memory_channels', 32),
    )

    if use_ae:
        model = wrap_model_with_autoencoder(
            lno_model=lno,
            full_grid_size=config['grid_size'],
            latent_grid_size=latent_grid,
            ndim=config.get('ae_ndim', 1),
            base_channels=config.get('ae_base_channels', 32),
            num_levels=config.get('ae_num_levels', None),
            max_channels=config.get('ae_max_channels', 256),
            activation=config.get('activation', 'swish'),
            lambda_recon=config.get('ae_lambda_recon', 0.1),
        )
        stats = compute_compression_stats(
            config['grid_size'],
            model.num_levels,
            config.get('ae_ndim', 1))
        print(f"Autoencoder: {config['grid_size']} → {latent_grid} "
              f"({stats['compression_ratio']:.0f}× compression, "
              f"{stats['memory_reduction_pct']:.1f}% memory reduction)")
        return model
    else:
        return lno


# ---------------------------------------------------------------------------
# Step 1: Generate data
# ---------------------------------------------------------------------------

def step_generate(config: dict):
    print("\n" + "=" * 60)
    print("STEP 1 / 3 : Generating Training Data")
    print("=" * 60)

    train_data, val_data, train_trajectories, val_trajectories = generate_training_data(
        grid_size=config['grid_size'],
        dx=config['dx'],
        dt=config['dt'],
        n_train=config['n_train'],
        n_val=config['n_val'],
        timestep_jump=config['timestep_jump'],
        trajectory_total_jumps=config.get('trajectory_total_jumps', 200),
        fdm_backend=config.get('fdm_backend', 'auto'),
        save_path='data/training_data.pt',
    )
    return train_data, val_data, train_trajectories, val_trajectories


# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------

def step_train(config: dict, train_data=None, val_data=None,
               train_trajectories=None, val_trajectories=None):
    print("\n" + "=" * 60)
    print("STEP 2 / 3 : Training Model")
    print("=" * 60)

    # Load data from disk if not passed directly
    if train_data is None or val_data is None:
        data_path = 'data/training_data.pt'
        if not os.path.exists(data_path):
            print(f"ERROR: {data_path} not found. Run --mode generate first.")
            sys.exit(1)
        data = torch.load(data_path, weights_only=False)
        train_data, val_data = data['train'], data['val']
        train_trajectories = data.get('train_trajectories')
        val_trajectories = data.get('val_trajectories')
    elif train_trajectories is None or val_trajectories is None:
        # Pair data passed directly but trajectories not provided — try disk
        data_path = 'data/training_data.pt'
        if os.path.exists(data_path):
            data = torch.load(data_path, weights_only=False)
            if train_trajectories is None:
                train_trajectories = data.get('train_trajectories')
            if val_trajectories is None:
                val_trajectories = data.get('val_trajectories')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model = _build_model(config)
    if not config.get('no_compile', False) and hasattr(torch, 'compile'):
        try:
            import triton  # noqa: F401 – inductor backend requires triton
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except (ImportError, Exception) as e:
            print(f"torch.compile skipped ({e})")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    config['total_params'] = total_params  # make visible in WandB sweep table
    if config.get('use_wandb') and WANDB_AVAILABLE:
        wandb.config.update({'total_params': total_params}, allow_val_change=True)
    print(f"Activation: {config.get('activation', 'swish')}")
    print(f"Temporal hidden: {config.get('temporal_hidden', 16)}, "
          f"temporal layers: {config.get('num_temporal_layers', 2)}, "
          f"NO layers: {config.get('num_no_layers', 2)}")
    print(f"Spectral filter: {config.get('spectral_filter', 'exponential')}, "
          f"strength: {config.get('filter_strength', 1.0)}")
    print(f"Ghost cells: {config.get('use_ghost_cells', True)}")

    # ── Optional autoencoder pre-training ──
    ae_pretrain_epochs = config.get('ae_pretrain_epochs', 0)
    if config.get('use_autoencoder', False) and ae_pretrain_epochs > 0:
        print(f"\nPre-training autoencoder for {ae_pretrain_epochs} epochs...")
        # Gather temperature fields for pre-training
        T_fields = train_data['T_n']  # [N, grid_size]
        if hasattr(model, 'temp_encoder'):
            from autoencoder import SpatialAutoEncoder, AutoEncoderPreTrainer

            ae = SpatialAutoEncoder(
                field_channels=1,
                base_channels=config.get('ae_base_channels', 32),
                num_levels=model.num_levels,
                ndim=config.get('ae_ndim', 1),
                max_channels=config.get('ae_max_channels', 256),
                activation=config.get('activation', 'swish'),
            )
            pretrainer = AutoEncoderPreTrainer(ae, device=device)
            ae_history = pretrainer.train(
                T_fields, epochs=ae_pretrain_epochs,
                lr=config.get('lr', 1e-3), batch_size=config['batch_size'])
            # Transfer pre-trained weights to the wrapped model
            model.temp_encoder.load_state_dict(ae.encoder.state_dict())
            model.temp_decoder.load_state_dict(ae.decoder.state_dict())
            print(f"  AE pre-training complete, final loss: {ae_history[-1]:.6f}")

    trainer = CattaneoTrainer(model, config, device)

    use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.watch(model, log='all', log_freq=50)

    history = trainer.train(
        train_data, val_data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        checkpoint_dir='checkpoints',
        resume_from=config.get('resume'),
        wandb_log=use_wandb,
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
    )

    return model, history


# ---------------------------------------------------------------------------
# Step 3: Benchmark (all modes)
# ---------------------------------------------------------------------------

def step_benchmark(config: dict):
    """
    Delegates to run_benchmark.py which contains per-mode FDM comparisons.
    """
    print("\n" + "=" * 60)
    print("STEP 3 / 3 : Benchmarking Neural vs FDM")
    print("=" * 60)

    from run_benchmark import run_all

    # alpha and rho_cp are placeholders — _load() in run_benchmark.py
    # will override them from the material file to match training data.
    ns = argparse.Namespace(
        grid_size=config['grid_size'],
        dx=config['dx'],
        dt=config['dt'],
        tau=config.get('tau', 1e-9),
        alpha=1e-4,
        rho_cp=1e6,
        bc_left=100.0,
        bc_right=200.0,
        timestep_jump=config['timestep_jump'],
        num_steps=config.get('num_steps', 0),
        time_over_tau=config.get('time_over_tau', 50.0),
        spectral_filter=config.get('spectral_filter', 'None'),
        filter_strength=config.get('filter_strength', 1.0),
        checkpoint=config.get('checkpoint', 'checkpoints/best_model.pt'),
        fdm_backend=config.get('fdm_backend', 'auto'),
    )

    run_all(ns)


def step_benchmark_single(config: dict, mode: str):
    """Run a single benchmark mode."""
    from run_benchmark import (
        benchmark_rollout,
        benchmark_hybrid,
        benchmark_super_res,
        benchmark_frame_gen,
        benchmark_warm_start,
        benchmark_scaling,
        benchmark_scaling_evolution,
    )

    # alpha and rho_cp are placeholders — _load() in run_benchmark.py
    # will override them from the material file to match training data.
    ns = argparse.Namespace(
        grid_size=config['grid_size'],
        dx=config['dx'],
        dt=config['dt'],
        tau=config.get('tau', 1e-9),
        alpha=1e-4,
        rho_cp=1e6,
        bc_left=100.0,
        bc_right=200.0,
        timestep_jump=config['timestep_jump'],
        num_steps=config.get('num_steps', 0),
        time_over_tau=config.get('time_over_tau', 50.0),
        spectral_filter=config.get('spectral_filter', 'None'),
        filter_strength=config.get('filter_strength', 1.0),
        checkpoint=config.get('checkpoint', 'checkpoints/best_model.pt'),
        fdm_backend=config.get('fdm_backend', 'auto'),
    )

    dispatch = {
        'rollout': benchmark_rollout,
        'hybrid': benchmark_hybrid,
        'super-res': benchmark_super_res,
        'frame-gen': benchmark_frame_gen,
        'warm-start': benchmark_warm_start,
        'scaling': benchmark_scaling,
        'scaling-evolution': benchmark_scaling_evolution,
    }
    dispatch[mode](ns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cattaneo-LNO: Generate -> Train -> Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow examples:

  # Full pipeline (generate data, train, benchmark all modes)
  python main.py --mode all

  # Individual steps
  python main.py --mode generate
  python main.py --mode train --wandb --epochs 200
  python main.py --mode benchmark

  # Run a single inference-mode benchmark
    python main.py --mode rollout
  python main.py --mode hybrid
  python main.py --mode super-res
  python main.py --mode frame-gen
  python main.py --mode warm-start
        """,
    )

    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'generate', 'train', 'benchmark',
                                 'rollout', 'hybrid', 'super-res', 'frame-gen',
                                 'warm-start', 'scaling', 'scaling-evolution'],
                        help='Execution mode')

    # Grid & physics
    parser.add_argument('--grid_size', type=int, default=112)
    parser.add_argument('--dx', type=float, default=1e-8)
    parser.add_argument('--dt', type=float, default=1e-13)
    parser.add_argument('--tau', type=float, default=1e-9)
    parser.add_argument('--timestep_jump', type=int, default=200)

    # Data generation
    parser.add_argument('--n_train', type=int, default=3000)
    parser.add_argument('--n_val', type=int, default=600)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=80,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    # Architecture
    parser.add_argument('--modes', type=int, default=16, help='Laplace poles')
    parser.add_argument('--width', type=int, default=64, help='Hidden width for the operator backbone')
    parser.add_argument('--num_no_layers', type=int, default=4,
                        help='Number of multi-scale operator blocks')
    parser.add_argument('--temporal_hidden', type=int, default=64,
                        help='Legacy temporal hidden size kept for checkpoint compatibility')
    parser.add_argument('--num_temporal_layers', type=int, default=2,
                        help='Legacy temporal layer count kept for checkpoint compatibility')
    parser.add_argument('--history_len', type=int, default=4,
                        help='Number of history frames used by the mandatory temporal encoder')
    parser.add_argument('--temporal_channels', type=int, default=32,
                        help='Channel width of the temporal convolution encoder')
    parser.add_argument('--local_conv_layers', type=int, default=2,
                        help='Number of dilated convolution layers in the local spatial path')
    parser.add_argument('--num_corrections', type=int, default=5,
                        help='Number of iterative correction steps applied after the velocity update')
    parser.add_argument('--use_recurrent_memory', action='store_true',
                        help='Enable Conv1d-GRU recurrent memory between autoregressive steps')
    parser.add_argument('--memory_channels', type=int, default=32,
                        help='Hidden channel width of the recurrent memory GRU cell')
    parser.add_argument('--activation', type=str, default='swish',
                        choices=['swish', 'mish', 'gelu', 'tanh'],
                        help='Activation function used in LNO blocks')

    # Spectral filtering
    parser.add_argument('--spectral_filter', type=str, default='exponential',
                        choices=['none', 'exponential', 'raised_cosine',
                                 'learnable', 'sharp_cutoff', 'transient_optimized',
                                 'dealias'])
    parser.add_argument('--filter_strength', type=float, default=1.0)

    # Ghost cells
    parser.add_argument('--no_ghost_cells', action='store_true',
                        help='Disable ghost boundary cells')

    # BPTT stabilisation
    parser.add_argument('--use_spectral_norm', action='store_true',
                        help='Apply spectral normalisation to pointwise convs in LNO blocks')
    parser.add_argument('--gradient_noise_std', type=float, default=0.0,
                        help='Std-dev of Gaussian noise injected into gradients after rollout-loss backward')
    parser.add_argument('--rollout_physics_weight', type=float, default=0.0,
                        help='Weight for per-step Cattaneo physics residual during rollout loss')
    parser.add_argument('--input_noise_std', type=float, default=0.003,
                        help='Std-dev of Gaussian noise (in dimensionless T*) added to inputs during training')

    # Polar weights / data-dependent poles / causal mask (sweepable)
    parser.add_argument('--max_amp', type=float, default=1.0,
                        help='Maximum amplitude for polar weight parameterisation')
    parser.add_argument('--amp_sharpness', type=float, default=1.0,
                        help='Sigmoid sharpness for amplitude bounding')
    parser.add_argument('--pole_offset_scale', type=float, default=0.1,
                        help='Scale for data-dependent pole offsets')
    parser.add_argument('--pole_min', type=float, default=0.1,
                        help='Minimum clamped pole value')
    parser.add_argument('--pole_max', type=float, default=100.0,
                        help='Maximum clamped pole value')
    parser.add_argument('--use_causal_mask', action='store_true', dest='use_causal_mask',
                        help='Enable data-dependent causal masking')
    parser.add_argument('--no_causal_mask', action='store_false', dest='use_causal_mask',
                        help='Disable data-dependent causal masking')
    parser.set_defaults(use_causal_mask=False)
    parser.add_argument('--causal_safety', type=float, default=1.0,
                        help='Safety multiplier for causal mask distance')
    parser.add_argument('--num_internal_steps', type=int, default=1,
                        help='Number of internal substeps per macro timestep (time integrator)')

    # Benchmark
    parser.add_argument('--num_steps', type=int, default=0,
                        help='Neural steps for benchmark rollouts')
    parser.add_argument('--time_over_tau', type=float, default=50.0,
                        help='Benchmark physical horizon in units of tau when num_steps <= 0')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Model checkpoint for benchmarking')
    parser.add_argument('--fdm_backend', type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu', 'newton'],
                        help='Backend for FDM reference solves and generation')

    # WandB
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='cattaneo-lno')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Adaptive Weighting arguments
    parser.add_argument('--adaptive_weighting.enabled', action='store_true', dest='adaptive_weighting_enabled')
    parser.add_argument('--adaptive_weighting.strategy', type=str, default='magnitude')
    parser.add_argument('--adaptive_weighting.warmup_epochs', type=int, default=10)
    parser.add_argument('--adaptive_weighting.temperature', type=float, default=1.0)
    parser.add_argument('--adaptive_weighting.ema_decay', type=float, default=0.9)
    parser.add_argument('--adaptive_weighting.update_frequency', type=int, default=10)

    # Rollout training - OPTIMIZED: quadratic warmup, compute every 2nd batch
    parser.add_argument('--rollout_steps_max', type=int, default=50,
                        help='Max autoregressive rollout steps for GT rollout loss')
    parser.add_argument('--rollout_steps_min', type=int, default=3,
                        help='Min rollout steps at start of training')
    parser.add_argument('--rollout_warmup_epochs', type=int, default=15,
                        help='Epochs to ramp rollout steps and lambda (quadratic warmup)')
    parser.add_argument('--lambda_rollout', type=float, default=2.0,
                        help='Target weight for GT rollout loss (reached after warmup)')
    parser.add_argument('--trajectory_total_jumps', type=int, default=200,
                        help='Total FDM jumps per trajectory sequence')
    parser.add_argument('--scheduled_sampling_start', type=int, default=0,
                        help='Epoch to start decaying teacher forcing')
    parser.add_argument('--scheduled_sampling_end', type=int, default=0,
                        help='Epoch when teacher forcing reaches 0 (0=no teacher forcing)')
    parser.add_argument('--rollout_every_n_batches', type=int, default=1,
                        help='Compute rollout loss every N batches to reduce iteration cost')

    # Physics loss weights
    parser.add_argument('--lambda_cattaneo', type=float, default=1.0,
                        help='Weight for Cattaneo PDE residual loss')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                        help='Weight for energy conservation loss')
    parser.add_argument('--lambda_bc', type=float, default=0.0,
                        help='Weight for boundary condition loss (0 when hard BCs are enforced)')
    parser.add_argument('--lambda_dTdt', type=float, default=0.01,
                        help='Weight for dTdt auxiliary head loss')
    parser.add_argument('--data_loss_floor_k', type=float, default=1e-3,
                        help='Absolute temperature-increment floor [K] used to stabilise tiny-step data loss')
    parser.add_argument('--lambda_gain', type=float, default=1.0,
                        help='Weight for gain-ratio penalty (drives gain toward 1.0x)')

    # Anti-drift losses
    parser.add_argument('--lambda_steady_state', type=float, default=0.1,
                        help='Weight for steady-state fixed-point loss (model should predict zero change at equilibrium)')
    parser.add_argument('--lambda_contraction', type=float, default=0.2,
                        help='Weight for contraction loss during rollout (penalises moving away from steady state)')
    parser.add_argument('--steady_state_every_n_batches', type=int, default=4,
                        help='Compute steady-state loss every N batches to reduce iteration cost')

    # Physics warmup
    parser.add_argument('--physics_warmup_epochs', type=int, default=10,
                        help='Epochs to ramp up physics loss from 0 to full weight')

    # Computational optimisations
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile (enabled by default on PyTorch 2.0+)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (simulates larger effective batch size)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes (0=main process)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'cosine_warm_restarts', 'one_cycle', 'reduce_on_plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--validation_max_batches', type=int, default=0,
                        help='Cap single-step validation to N batches per epoch (0=full validation)')
    parser.add_argument('--train_eval_max_batches', type=int, default=4,
                        help='Evaluate N clean training batches per epoch for a train/val comparable metric (0=disable)')
    parser.add_argument('--val_rollout_every_n_epochs', type=int, default=1,
                        help='Run rollout validation every N epochs instead of every epoch')
    parser.add_argument('--val_rollout_max_batches', type=int, default=8,
                        help='Cap rollout validation to N trajectory batches when it runs')
    parser.add_argument('--stability_m_steps', type=int, default=24,
                        help='No-grad rollout horizon used by the optional stability critic')
    parser.add_argument('--lambda_stability', type=float, default=0.0,
                        help='Weight for the optional stability critic (0 disables it)')
    parser.add_argument('--stability_loss_every_n_batches', type=int, default=8,
                        help='Compute clean-state stability loss every N training batches')
    parser.add_argument('--stability_traj_loss_every_n_batches', type=int, default=16,
                        help='Compute trajectory-state stability loss every N rollout batches')

    # Autoencoder (spatial compression for large grids)
    parser.add_argument('--use_autoencoder', action='store_true',
                        help='Enable spatial autoencoder for large-grid compression')
    parser.add_argument('--ae_latent_grid', type=int, default=None,
                        help='Latent grid size (default: grid_size // 4)')
    parser.add_argument('--ae_ndim', type=int, default=1, choices=[1, 2, 3],
                        help='Spatial dimensionality (1D, 2D, or 3D)')
    parser.add_argument('--ae_base_channels', type=int, default=32,
                        help='Autoencoder base channel width')
    parser.add_argument('--ae_num_levels', type=int, default=None,
                        help='Autoencoder downsampling levels (auto-computed if not set)')
    parser.add_argument('--ae_max_channels', type=int, default=256,
                        help='Max autoencoder channel width (reduce for 3D)')
    parser.add_argument('--ae_lambda_recon', type=float, default=0.1,
                        help='Weight for autoencoder reconstruction loss')
    parser.add_argument('--ae_pretrain_epochs', type=int, default=0,
                        help='Pre-train autoencoder for N epochs before full training (0=skip)')

    args = parser.parse_args()

    # -- Build config dict --
    config = dict(
        grid_size=args.grid_size,
        dx=args.dx,
        dt=args.dt,
        tau=args.tau,
        T_ref=200.0,
        delta_T=100.0,
        alpha_ref=_get_alpha_ref(),
        n_train=args.n_train,
        n_val=args.n_val,
        timestep_jump=args.timestep_jump,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        modes=args.modes,
        width=args.width,
        num_no_layers=args.num_no_layers,
        temporal_hidden=args.temporal_hidden,
        num_temporal_layers=args.num_temporal_layers,
        history_len=args.history_len,
        temporal_channels=args.temporal_channels,
        local_conv_layers=args.local_conv_layers,
        num_corrections=args.num_corrections,
        activation=args.activation,
        weight_decay=1e-4,
        lambda_cattaneo=args.lambda_cattaneo,
        lambda_energy=args.lambda_energy,
        lambda_characteristic=0.0,
        lambda_bc=args.lambda_bc,
        lambda_dTdt=args.lambda_dTdt,
        data_loss_floor_k=args.data_loss_floor_k,
        lambda_gain=args.lambda_gain,
        patience=args.patience,
        spectral_filter=args.spectral_filter,
        filter_strength=args.filter_strength,
        use_ghost_cells=not args.no_ghost_cells,
        use_wandb=args.wandb,
        resume=args.resume,
        num_steps=args.num_steps,
        time_over_tau=args.time_over_tau,
        checkpoint=args.checkpoint,
        fdm_backend=args.fdm_backend,
        
        # Adaptive weighting (keys contain dots; use vars(args) to access safely)
        **{
            'adaptive_weighting.enabled': args.adaptive_weighting_enabled,
            'adaptive_weighting.strategy': vars(args).get('adaptive_weighting.strategy', 'magnitude'),
            'adaptive_weighting.warmup_epochs': vars(args).get('adaptive_weighting.warmup_epochs', 10),
            'adaptive_weighting.temperature': vars(args).get('adaptive_weighting.temperature', 1.0),
            'adaptive_weighting.ema_decay': vars(args).get('adaptive_weighting.ema_decay', 0.9),
            'adaptive_weighting.update_frequency': vars(args).get('adaptive_weighting.update_frequency', 10),
        },

        # Rollout training
        rollout_steps_max=args.rollout_steps_max,
        rollout_steps_min=args.rollout_steps_min,
        rollout_warmup_epochs=args.rollout_warmup_epochs,
        lambda_rollout=args.lambda_rollout,
        trajectory_total_jumps=args.trajectory_total_jumps,
        scheduled_sampling_start=args.scheduled_sampling_start,
        scheduled_sampling_end=args.scheduled_sampling_end,
        rollout_every_n_batches=args.rollout_every_n_batches,

        # Anti-drift losses
        lambda_steady_state=args.lambda_steady_state,
        lambda_contraction=args.lambda_contraction,
        steady_state_every_n_batches=args.steady_state_every_n_batches,

        # BPTT stabilisation
        use_spectral_norm=args.use_spectral_norm,
        gradient_noise_std=args.gradient_noise_std,
        rollout_physics_weight=args.rollout_physics_weight,
        input_noise_std=args.input_noise_std,

        # Polar weights / data-dependent poles / causal mask
        max_amp=args.max_amp,
        amp_sharpness=args.amp_sharpness,
        pole_offset_scale=args.pole_offset_scale,
        pole_min=args.pole_min,
        pole_max=args.pole_max,
        use_causal_mask=args.use_causal_mask,
        causal_safety=args.causal_safety,
        num_internal_steps=args.num_internal_steps,

        # Recurrent memory
        use_recurrent_memory=args.use_recurrent_memory,
        memory_channels=args.memory_channels,

        # Physics warmup and computational optimisations
        physics_warmup_epochs=args.physics_warmup_epochs,
        curriculum_data_only_frac=getattr(args, 'curriculum_data_only_frac', 0.20),
        curriculum_mixed_frac=getattr(args, 'curriculum_mixed_frac', 0.30),
        grad_balance_every=getattr(args, 'grad_balance_every', 5),
        no_compile=args.no_compile,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        scheduler=args.scheduler,
        validation_max_batches=args.validation_max_batches,
        train_eval_max_batches=args.train_eval_max_batches,
        val_rollout_every_n_epochs=args.val_rollout_every_n_epochs,
        val_rollout_max_batches=args.val_rollout_max_batches,
        stability_m_steps=args.stability_m_steps,
        lambda_stability=args.lambda_stability,
        stability_loss_every_n_batches=args.stability_loss_every_n_batches,
        stability_traj_loss_every_n_batches=args.stability_traj_loss_every_n_batches,

        # Autoencoder (spatial compression)
        use_autoencoder=args.use_autoencoder,
        ae_latent_grid=args.ae_latent_grid or (args.grid_size // 4),
        ae_ndim=args.ae_ndim,
        ae_base_channels=args.ae_base_channels,
        ae_num_levels=args.ae_num_levels,
        ae_max_channels=args.ae_max_channels,
        ae_lambda_recon=args.ae_lambda_recon,
        ae_pretrain_epochs=args.ae_pretrain_epochs,
    )

    # -- WandB init --
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed.  pip install wandb")
            config['use_wandb'] = False
        else:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                       config=config)
            print(f"WandB initialised -> project={args.wandb_project}")

    setup_directories()

    # -- Dispatch --
    mode = args.mode

    if mode == 'all':
        train_data, val_data, train_traj, val_traj = step_generate(config)
        step_train(config, train_data, val_data, train_traj, val_traj)
        step_benchmark(config)

    elif mode == 'generate':
        step_generate(config)

    elif mode == 'train':
        step_train(config)

    elif mode == 'benchmark':
        step_benchmark(config)

    elif mode in ('rollout', 'hybrid', 'super-res', 'frame-gen',
                  'warm-start', 'scaling', 'scaling-evolution'):
        step_benchmark_single(config, mode)

    # -- Finish WandB --
    if config.get('use_wandb') and WANDB_AVAILABLE:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
