"""
Evaluation and Visualization Tools for Cattaneo-LNO
====================================================

Provides tools for:
- Model evaluation on test cases
- Comparison with FDM baseline
- Visualization of predictions and errors
- Performance benchmarking
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time
import json
from dataclasses import dataclass

from cattaneo_lno import create_cattaneo_model
from hybrid_solver import HybridCattaneoSolver, HybridSolverConfig


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    mse: float
    rmse: float
    mae: float
    relative_error: float
    max_error: float
    inference_time: float
    method_breakdown: Optional[Dict] = None


def load_model(checkpoint_path: str, grid_size: int, device: str = 'cuda',
               dx: float = 1e-8, spectral_filter: str = 'exponential',
               filter_strength: float = 4.0, use_ghost_cells: bool = True):
    """Load trained model from checkpoint.
    
    Architecture and forward-pass hyperparameters (spectral_filter,
    filter_strength, max_amp, amp_sharpness, pole_*, causal_*)
    are restored from the checkpoint config when available, so the
    model is reconstructed exactly as it was during training.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint.get('model_state_dict', {})
    if state_dict and any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {
            (k[len('_orig_mod.'): ] if k.startswith('_orig_mod.') else k): v
            for k, v in state_dict.items()
        }

    # Recover optional recurrent-memory branch from config when present,
    # otherwise infer it from the checkpoint weights for compatibility.
    use_recurrent_memory = config.get(
        'use_recurrent_memory',
        any(k.startswith('predictor.memory_cell.') for k in state_dict.keys())
    )
    memory_channels = config.get('memory_channels')
    if memory_channels is None:
        if 'predictor.memory_cell.W_h.weight' in state_dict:
            memory_channels = state_dict['predictor.memory_cell.W_h.weight'].shape[0]
        elif 'predictor.memory_cell.W_rz.weight' in state_dict:
            memory_channels = state_dict['predictor.memory_cell.W_rz.weight'].shape[0] // 2
        elif 'predictor.memory_fusion.proj.weight' in state_dict:
            memory_channels = state_dict['predictor.memory_fusion.proj.weight'].shape[1]
        else:
            memory_channels = 32

    # Detect whether the checkpoint was saved with spectral normalisation enabled.
    # spectral_norm replaces 'pointwise.weight' with 'pointwise.weight_orig/_u/_v'.
    use_spectral_norm = any(
        'pointwise.weight_orig' in k for k in state_dict.keys()
    )

    # Recover architecture from checkpoint weights when config lacks it.
    input_proj_w = state_dict.get('input_proj.weight', None)
    width = config.get('width', input_proj_w.shape[0] if input_proj_w is not None else 64)

    # Detect whether this is a new-style (SecondOrderPredictor) checkpoint
    # or a legacy (DualStatePredictor) checkpoint.
    is_new_arch = any(k.startswith('predictor.blocks.') for k in state_dict.keys())

    if is_new_arch:
        # New architecture: predictor.blocks.N.spectral_conv...
        block_ids = set()
        for key in state_dict.keys():
            if key.startswith('predictor.blocks.'):
                parts = key.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    block_ids.add(int(parts[2]))
        num_no_layers = config.get('num_no_layers', max(block_ids) + 1 if block_ids else 4)
    else:
        # Legacy: dual_predictor.no_layers.N...
        no_layer_ids = set()
        for key in state_dict.keys():
            if key.startswith('dual_predictor.no_layers.'):
                parts = key.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    no_layer_ids.add(int(parts[2]))
        num_no_layers = config.get('num_no_layers', max(no_layer_ids) + 1 if no_layer_ids else 4)

    if 'dual_predictor.temporal_encoder.gru.weight_hh_l0' in state_dict:
        temporal_hidden = state_dict['dual_predictor.temporal_encoder.gru.weight_hh_l0'].shape[1]
    else:
        temporal_hidden = config.get('temporal_hidden', 64)

    temporal_layer_ids = set()
    for key in state_dict.keys():
        if key.startswith('dual_predictor.temporal_encoder.gru.weight_ih_l'):
            suffix = key.rsplit('l', 1)[-1]
            if suffix.isdigit():
                temporal_layer_ids.add(int(suffix))
    num_temporal_layers = config.get(
        'num_temporal_layers',
        max(temporal_layer_ids) + 1 if temporal_layer_ids else 2
    )

    # Recover modes (number of Laplace poles) from log_poles shape
    modes_default = 16
    for key in state_dict.keys():
        if key.endswith('.log_poles'):
            modes_default = state_dict[key].shape[0]
            break
    modes = config.get('modes', modes_default)

    # Infer extended_grid from the checkpoint to reconstruct a matching
    # architecture regardless of how use_ghost_cells has changed since training.
    ckpt_extended_grid = None
    if 'wave_speed_net.0.weight' in state_dict:
        ckpt_extended_grid = state_dict['wave_speed_net.0.weight'].shape[1]
    elif 'dual_predictor.temporal_encoder.gru.weight_ih_l0' in state_dict:
        ckpt_extended_grid = state_dict['dual_predictor.temporal_encoder.gru.weight_ih_l0'].shape[1]

    if ckpt_extended_grid is not None:
        if grid_size == ckpt_extended_grid:
            # Checkpoint was trained without ghost cells
            use_ghost_cells = False
        elif grid_size + 2 == ckpt_extended_grid:
            # Checkpoint was trained with ghost cells
            use_ghost_cells = True
        else:
            # Grid size mismatch: infer grid_size from checkpoint, prefer ghost cells
            if ckpt_extended_grid >= 2:
                grid_size = ckpt_extended_grid - 2
                use_ghost_cells = True
            else:
                grid_size = ckpt_extended_grid
                use_ghost_cells = False

    L = grid_size * dx
    model = create_cattaneo_model(
        grid_size=grid_size,
        modes=modes,
        width=width,
        num_no_layers=num_no_layers,
        temporal_hidden=temporal_hidden,
        num_temporal_layers=num_temporal_layers,
        timestep_jump=config.get('timestep_jump', 1),
        activation=config.get('activation', 'swish'),
        L=L,
        alpha_ref=config.get('alpha_ref', 1e-4),
        T_ref=config.get('T_ref', 200.0),
        delta_T=config.get('delta_T', 100.0),
        tau_ref=config.get('tau_ref', 1e-9),
        spectral_filter=config.get('spectral_filter', spectral_filter),
        filter_strength=config.get('filter_strength', filter_strength),
        use_ghost_cells=use_ghost_cells,
        use_spectral_norm=use_spectral_norm,
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
        use_recurrent_memory=use_recurrent_memory,
        memory_channels=memory_channels,
    )

    try:
        # Handle channel-count mismatches between checkpoint and current model.
        # Previously the model used 8 input channels (incl. dt_star); now it
        # uses 7.  Pad or truncate the input_proj weight accordingly.
        ip_key = 'input_proj.weight'
        if ip_key in state_dict:
            ckpt_ch = state_dict[ip_key].shape[1]
            model_ch = model.input_proj.weight.shape[1]
            if ckpt_ch != model_ch:
                old_w = state_dict[ip_key]
                new_w = torch.zeros(old_w.shape[0], model_ch, old_w.shape[2],
                                    dtype=old_w.dtype, device=old_w.device)
                copy_ch = min(ckpt_ch, model_ch)
                new_w[:, :copy_ch, :] = old_w[:, :copy_ch, :]
                state_dict[ip_key] = new_w

        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(str(e))
    
    model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, test_data: Dict, device: str = 'cuda') -> EvaluationResult:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_data: Test dataset
        device: Device to use
        
    Returns:
        EvaluationResult with metrics
    """
    model.eval()
    
    # Move data to device
    T_n = test_data['T_n'].to(device)
    T_nm1 = test_data['T_nm1'].to(device)
    T_target = test_data['T_target'].to(device)
    q = test_data['q'].to(device)
    tau = test_data['tau'].to(device)
    alpha = test_data['alpha'].to(device)
    rho_cp = test_data['rho_cp'].to(device)
    bc_left = test_data['bc_left'].to(device)
    bc_right = test_data['bc_right'].to(device)
    dt = test_data['dt']
    dx = test_data['dx']
    
    # Inference
    start_time = time.time()
    
    with torch.no_grad():
        predictions = model(T_n, T_nm1, q, tau, alpha, rho_cp,
                          bc_left, bc_right, dt, dx)
        T_pred = predictions['T_pred']
    
    inference_time = time.time() - start_time
    
    # Compute metrics
    diff = T_pred - T_target
    abs_diff = diff.abs()
    mse = torch.mean(diff ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(abs_diff).item()

    # Relative error
    target_norm = torch.norm(T_target)
    relative_error = torch.norm(diff) / (target_norm + 1e-10)
    relative_error = relative_error.item() * 100  # Percentage

    max_error = torch.max(abs_diff).item()
    
    return EvaluationResult(
        mse=mse,
        rmse=rmse,
        mae=mae,
        relative_error=relative_error,
        max_error=max_error,
        inference_time=inference_time
    )


def compare_with_fdm(model, test_case: Dict, grid: np.ndarray,
                    tau: float, num_steps: int, device: str = 'cuda') -> Dict:
    """
    Compare neural model with FDM baseline.
    
    Args:
        model: Neural model
        test_case: Test case parameters
        grid: Material grid
        tau: Relaxation time
        num_steps: Number of steps
        device: Device
        
    Returns:
        Comparison results
    """
    from HF_Cattaneo import nl_solve_HF_1d_Cattaneo
    
    grid_size = len(test_case['T0'])
    dx = test_case['dx']
    dt = test_case['dt']
    alpha = test_case['alpha']
    rho_cp = test_case['rho_cp']
    
    # Neural prediction
    config = HybridSolverConfig(
        residual_threshold=2.0,
        gradient_threshold=500.0,
        timestep_jump=getattr(model, 'timestep_jump', 1),
        max_timestep_jump=getattr(model, 'timestep_jump', 1),
        verbose=False
    )
    solver = HybridCattaneoSolver(grid_size, dx, dt, model, config)
    
    start_time = time.time()
    neural_result = solver.solve(
        test_case['T0'], grid, num_steps, tau,
        alpha, rho_cp,
        bc_left=test_case['bc_left'],
        bc_right=test_case['bc_right'],
        k_temp_dependent=False, c_temp_dependent=False,
        save_history=True
    )
    neural_time = time.time() - start_time
    
    # FDM baseline (constant properties for fair comparison)
    # Use the actual number of effective FDM-equivalent steps from the hybrid solver
    # so both solutions reach the same physical time.
    total_fdm_steps = neural_result.get('total_effective_fdm_steps', num_steps * solver.current_jump)
    
    T = test_case['T0'].copy()
    T_prev = test_case['T0'].copy()
    
    start_time = time.time()
    fdm_history = [T.copy()]
    
    for _ in range(total_fdm_steps):
        T_new, _, _ = nl_solve_HF_1d_Cattaneo(
            grid_size, grid, T, T_prev,
            dx=dx, dt=dt, tau=tau,
            tol=1e-6, max_newton_iters=100,
            k_temp_dependent=False, c_temp_dependent=False,
            verbose=False,
            BC=(test_case['bc_left'], test_case['bc_right'])
        )
        T_prev = T.copy()
        T = T_new
        fdm_history.append(T.copy())
    
    fdm_time = time.time() - start_time
    
    # Compute errors
    neural_final = neural_result['T_final']
    fdm_final = fdm_history[-1]
    
    error_vs_fdm = np.linalg.norm(neural_final - fdm_final) / np.linalg.norm(fdm_final) * 100
    
    # Interior-only error (excluding boundary nodes affected by hard BC vs ghost-node mismatch)
    n_bc = 5
    interior_error = np.linalg.norm(neural_final[n_bc:-n_bc] - fdm_final[n_bc:-n_bc]) / (np.linalg.norm(fdm_final[n_bc:-n_bc]) + 1e-15) * 100
    
    return {
        'neural_result': neural_result,
        'fdm_history': np.array(fdm_history),
        'neural_time': neural_time,
        'fdm_time': fdm_time,
        'fdm_steps': total_fdm_steps,
        'speedup': fdm_time / neural_time,
        'error_vs_fdm_percent': error_vs_fdm,
        'interior_error_percent': interior_error,
        'neural_final': neural_final,
        'fdm_final': fdm_final
    }


def plot_comparison(comparison: Dict, save_path: Optional[str] = None):
    """
    Plot comparison between neural and FDM.
    
    Args:
        comparison: Output from compare_with_fdm
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    neural_history = comparison['neural_result'].get('T_history', None)
    fdm_history = comparison['fdm_history']
    
    # Plot 1: Final temperature profiles
    ax = axes[0, 0]
    x = np.arange(len(comparison['neural_final']))
    ax.plot(x, comparison['fdm_final'], 'b-', linewidth=2, label='FDM')
    ax.plot(x, comparison['neural_final'], 'r--', linewidth=2, label='Neural')
    ax.set_xlabel('Grid Point')
    ax.set_ylabel('Temperature')
    ax.set_title('Final Temperature Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Absolute error
    ax = axes[0, 1]
    error = np.abs(comparison['neural_final'] - comparison['fdm_final'])
    ax.semilogy(x, error, 'g-', linewidth=2)
    ax.set_xlabel('Grid Point')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Error Distribution (Rel: {comparison["error_vs_fdm_percent"]:.3f}%)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Temporal evolution (if available)
    ax = axes[1, 0]
    if neural_history is not None and len(neural_history) > 0:
        # Plot every Nth timestep
        step = max(1, len(neural_history) // 10)
        for i in range(0, len(neural_history), step):
            alpha_val = 0.3 + 0.7 * (i / len(neural_history))
            ax.plot(x, neural_history[i], 'r-', alpha=alpha_val, linewidth=1)
        ax.set_xlabel('Grid Point')
        ax.set_ylabel('Temperature')
        ax.set_title('Neural: Temporal Evolution')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'History not saved', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Neural: Temporal Evolution')
    
    # Plot 4: Performance metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics_text = f"""
    Performance Metrics:
    
    FDM Time: {comparison['fdm_time']:.3f}s
    Neural Time: {comparison['neural_time']:.3f}s
    Speedup: {comparison['speedup']:.2f}x
    
    Accuracy:
    Relative Error vs FDM: {comparison['error_vs_fdm_percent']:.4f}%
    
    Neural Method Breakdown:
    Neural %: {comparison['neural_result'].get('neural_percentage', 0):.1f}%
    """
    
    ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history_path: str, save_path: Optional[str] = None):
    """
    Plot training history from saved JSON.
    
    Args:
        history_path: Path to history JSON file
        save_path: Optional path to save figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Total loss
    ax = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.semilogy(epochs, history['train_loss'], 'b-', label='Train')
    ax.semilogy(epochs, history['val_loss'], 'r-', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training History: Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss breakdown
    ax = axes[1]
    ax.semilogy(epochs, history['train_data_loss'], 'b-', label='Train Data')
    ax.semilogy(epochs, history['train_physics_loss'], 'b--', label='Train Physics')
    ax.semilogy(epochs, history['val_data_loss'], 'r-', label='Val Data')
    ax.semilogy(epochs, history['val_physics_loss'], 'r--', label='Val Physics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History: Loss Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def benchmark_model(model, test_cases: List[Dict], device: str = 'cuda') -> Dict:
    """
    Benchmark model on multiple test cases.
    
    Args:
        model: Neural model
        test_cases: List of test case dictionaries
        device: Device
        
    Returns:
        Benchmark results
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"Benchmarking case {i+1}/{len(test_cases)}...")
        
        # Create test data dict
        batch_size = 1
        grid_size = len(test_case['T0'])
        
        test_data = {
            'T_n': torch.from_numpy(test_case['T0']).float().unsqueeze(0),
            'T_nm1': torch.from_numpy(test_case['T0']).float().unsqueeze(0),
            'T_target': torch.from_numpy(test_case['T_target']).float().unsqueeze(0),
            'q': torch.zeros(1, grid_size),
            'tau': torch.full((1, grid_size), test_case['tau']),
            'alpha': torch.from_numpy(test_case['alpha']).float().unsqueeze(0),
            'rho_cp': torch.from_numpy(test_case['rho_cp']).float().unsqueeze(0),
            'bc_left': torch.tensor([test_case['bc_left']]),
            'bc_right': torch.tensor([test_case['bc_right']]),
            'dt': test_case['dt'],
            'dx': test_case['dx']
        }
        
        result = evaluate_model(model, test_data, device)
        results.append(result)
    
    # Aggregate results
    avg_mse = np.mean([r.mse for r in results])
    avg_rmse = np.mean([r.rmse for r in results])
    avg_mae = np.mean([r.mae for r in results])
    avg_relative = np.mean([r.relative_error for r in results])
    avg_time = np.mean([r.inference_time for r in results])
    
    return {
        'individual_results': results,
        'avg_mse': avg_mse,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_relative_error_percent': avg_relative,
        'avg_inference_time': avg_time
    }


def generate_report(evaluation_results: Dict, save_path: str):
    """
    Generate evaluation report.
    
    Args:
        evaluation_results: Results from benchmark_model
        save_path: Path to save report
    """
    report = f"""
    Cattaneo-LNO Evaluation Report
    ================================
    
    Average Metrics:
    - MSE: {evaluation_results['avg_mse']:.6e}
    - RMSE: {evaluation_results['avg_rmse']:.6e}
    - MAE: {evaluation_results['avg_mae']:.6e}
    - Relative Error: {evaluation_results['avg_relative_error_percent']:.4f}%
    - Inference Time: {evaluation_results['avg_inference_time']:.4f}s
    
    Individual Results:
    """
    
    for i, result in enumerate(evaluation_results['individual_results']):
        report += f"""
    Case {i+1}:
    - MSE: {result.mse:.6e}
    - RMSE: {result.rmse:.6e}
    - Relative Error: {result.relative_error:.4f}%
    """
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Saved report to {save_path}")


if __name__ == '__main__':
    # Example usage
    print("Evaluation tools loaded.")
    print("\nExample usage:")
    print("  1. Load model: model, ckpt = load_model('checkpoints/best_model.pt', grid_size=112)")
    print("  2. Evaluate: result = evaluate_model(model, test_data)")
    print("  3. Compare: comp = compare_with_fdm(model, test_case, grid, tau, num_steps)")
    print("  4. Plot: plot_comparison(comp, 'comparison.png')")
