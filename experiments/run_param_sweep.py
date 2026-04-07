#!/usr/bin/env python
"""Automated parameter sweep for MLE registration.

Explores the relationship between hyperparameters and registration performance.
Generates timestamped logs, configs, and results for each experiment.

Usage:
    python experiments/run_param_sweep.py
"""

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import itertools
from typing import List, Dict, Any
import traceback

import taichi as ti

# Initialize Taichi
ti.init(arch=ti.cuda, log_level=ti.ERROR)
device = 'cuda'

from gmm_point_alignment.unified_registration import (
    UnifiedRegistration,
    UnifiedConfig,
    RegistrationMethod,
)
from gmm_point_alignment.transform_utils import se3_exp
from misc.hier_IO import load_hier_to_torch


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    # Parameters
    exp_id: str
    timestamp: str
    config: Dict[str, Any]

    # Results
    converged: bool
    final_loss: float
    translation_error: float
    rotation_error: float
    scale_error: float
    num_iters: int

    # Timing
    grid_build_time: float
    registration_time: float

    # Status
    success: bool
    error_message: str = ""


def setup_data():
    """Load and prepare test data."""
    # Load scene
    hier_path = Path('../data/merged.hier')
    hier_scene = load_hier_to_torch(hier_path, device=device)
    scene = hier_scene.gaussian_scene

    # Sample point cloud
    np.random.seed(42)
    sample_indices = torch.randperm(len(scene.position))[:1000]
    pointcloud = scene.position[sample_indices].clone()

    # Apply known transform
    xi_true = torch.tensor([0.3, -0.2, 0.1, 0.05, -0.03, 0.02], device=device)
    T_true = se3_exp(xi_true)
    R_true = T_true[:3, :3]
    t_true = T_true[:3, 3]
    pointcloud_transformed = (R_true @ pointcloud.T).T + t_true

    # Ground truth inverse
    T_true_inv = torch.inverse(T_true)

    return scene, pointcloud_transformed, T_true_inv


def compute_errors(result, T_true_inv):
    """Compute translation and rotation errors."""
    R_true_inv = T_true_inv[:3, :3]
    t_true_inv = T_true_inv[:3, 3]

    # Translation error
    t_error = (result.t - t_true_inv).norm().item()

    # Rotation error
    R_result = result.R
    R_rel = R_result.T @ R_true_inv
    cos_angle = torch.clamp((R_rel.trace() - 1) / 2, -1, 1)
    angle_error = torch.acos(cos_angle).item() * 180 / np.pi

    return t_error, angle_error


def run_single_experiment(
    exp_id: str,
    params: Dict[str, Any],
    scene,
    pointcloud,
    T_true_inv
) -> ExperimentResult:
    """Run a single experiment with given parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Create config
        config = UnifiedConfig(
            method=RegistrationMethod.MLE,
            mle_voxel_strategy=params.get('voxel_strategy', 'median_radius'),
            mle_voxel_factor=params.get('voxel_factor', 1.0),
            mle_num_iters=params.get('num_iters', 100),
            mle_lr=params.get('lr', 0.01),
            mle_lr_translation=params.get('lr_translation', 0.01),
            mle_lr_rotation=params.get('lr_rotation', 0.01),
            mle_use_scale=params.get('use_scale', False),
            mle_multi_init=params.get('multi_init', True),
            mle_num_init=params.get('num_init', 5),
            mle_debug=False,  # Disable debug plots for sweep
        )

        # Run registration
        import time
        t0 = time.time()
        reg = UnifiedRegistration(config)
        t1 = time.time()
        result = reg.register(scene, pointcloud)
        t2 = time.time()

        # Compute errors
        t_error, angle_error = compute_errors(result, T_true_inv)

        # Scale error (if applicable)
        if params.get('use_scale', False):
            scale_true = params.get('scale_true', 1.0)
            scale_error = abs(result.scale - scale_true)
        else:
            scale_error = 0.0

        return ExperimentResult(
            exp_id=exp_id,
            timestamp=timestamp,
            config=params,
            converged=result.converged,
            final_loss=result.error,
            translation_error=t_error,
            rotation_error=angle_error,
            scale_error=scale_error,
            num_iters=result.num_iters,
            grid_build_time=t1 - t0,
            registration_time=t2 - t1,
            success=True
        )

    except Exception as e:
        return ExperimentResult(
            exp_id=exp_id,
            timestamp=timestamp,
            config=params,
            converged=False,
            final_loss=float('inf'),
            translation_error=float('inf'),
            rotation_error=float('inf'),
            scale_error=float('inf'),
            num_iters=0,
            grid_build_time=0,
            registration_time=0,
            success=False,
            error_message=str(e) + "\n" + traceback.format_exc()
        )


def save_experiment(result: ExperimentResult, exp_dir: Path):
    """Save experiment results and config."""
    exp_subdir = exp_dir / f"{result.timestamp}_{result.exp_id}"
    exp_subdir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = exp_subdir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(result.config, f, indent=2)

    # Save results
    result_dict = {
        'exp_id': result.exp_id,
        'timestamp': result.timestamp,
        'converged': result.converged,
        'final_loss': result.final_loss,
        'translation_error': result.translation_error,
        'rotation_error': result.rotation_error,
        'scale_error': result.scale_error,
        'num_iters': result.num_iters,
        'grid_build_time': result.grid_build_time,
        'registration_time': result.registration_time,
        'success': result.success,
        'error_message': result.error_message if not result.success else "",
    }
    result_path = exp_subdir / "results.json"
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    return exp_subdir


def generate_summary(all_results: List[ExperimentResult], summary_path: Path):
    """Generate summary report of all experiments."""
    lines = []
    lines.append("=" * 100)
    lines.append("MLE Registration Parameter Sweep Summary")
    lines.append("=" * 100)
    lines.append(f"Total experiments: {len(all_results)}")
    lines.append(f"Successful: {sum(1 for r in all_results if r.success)}")
    lines.append(f"Failed: {sum(1 for r in all_results if not r.success)}")
    lines.append("")

    # Header
    lines.append(f"{'ExpID':<15} {'T-LR':<8} {'R-LR':<8} {'Iters':<6} {'Conv':<5} "
                 f"{'T-Err(m)':<10} {'R-Err(°)':<10} {'Time(s)':<10}")
    lines.append("-" * 100)

    # Sort by translation error
    sorted_results = sorted(
        [r for r in all_results if r.success],
        key=lambda x: x.translation_error + x.rotation_error
    )

    for r in sorted_results[:20]:  # Top 20
        lines.append(
            f"{r.exp_id:<15} "
            f"{r.config.get('lr_translation', 0.01):<8.4f} "
            f"{r.config.get('lr_rotation', 0.01):<8.4f} "
            f"{r.num_iters:<6} "
            f"{'Y' if r.converged else 'N':<5} "
            f"{r.translation_error:<10.4f} "
            f"{r.rotation_error:<10.2f} "
            f"{r.registration_time:<10.2f}"
        )

    lines.append("")
    lines.append("=" * 100)
    lines.append("Best Configuration (by combined error):")
    lines.append("=" * 100)

    if sorted_results:
        best = sorted_results[0]
        lines.append(f"Experiment: {best.exp_id}")
        lines.append(f"Config:")
        for k, v in best.config.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"")
        lines.append(f"Results:")
        lines.append(f"  Translation error: {best.translation_error:.4f} m")
        lines.append(f"  Rotation error: {best.rotation_error:.2f}°")
        lines.append(f"  Final loss: {best.final_loss:.4f}")
        lines.append(f"  Converged: {best.converged}")
        lines.append(f"  Registration time: {best.registration_time:.2f} s")

    # Write summary
    summary_text = "\n".join(lines)
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    print(summary_text)


def main():
    """Main parameter sweep."""
    print("=" * 70)
    print("MLE Registration Parameter Sweep")
    print("=" * 70)

    # Setup directories
    exp_dir = Path(__file__).parent
    logs_dir = exp_dir / "logs"
    configs_dir = exp_dir / "configs"
    results_dir = exp_dir / "results"

    for d in [logs_dir, configs_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup data
    print("\nLoading data...")
    scene, pointcloud, T_true_inv = setup_data()
    print(f"Scene: {len(scene.position)} spheres")
    print(f"Point cloud: {len(pointcloud)} points")

    # Define parameter grid
    param_grid = {
        'lr_translation': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'lr_rotation': [0.001, 0.005, 0.01, 0.02, 0.05],
        'num_iters': [100, 200],
        'multi_init': [True],
        'num_init': [5],
        'voxel_strategy': ['median_radius'],
        'voxel_factor': [1.0],
        'use_scale': [False],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    print(f"\nTotal experiments to run: {len(all_combinations)}")
    print("=" * 70)

    # Run experiments
    all_results = []
    for i, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        exp_id = f"exp_{i:04d}"

        print(f"\n[{i+1}/{len(all_combinations)}] Running {exp_id}...")
        print(f"  lr_t={params['lr_translation']:.4f}, lr_r={params['lr_rotation']:.4f}")

        result = run_single_experiment(exp_id, params, scene, pointcloud, T_true_inv)
        all_results.append(result)

        # Save immediately
        save_experiment(result, results_dir)

        if result.success:
            print(f"  ✓ T-err: {result.translation_error:.4f}m, "
                  f"R-err: {result.rotation_error:.2f}°, "
                  f"Loss: {result.final_loss:.2f}")
        else:
            print(f"  ✗ Failed: {result.error_message[:100]}")

    # Generate summary
    print("\n" + "=" * 70)
    print("Generating summary...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = logs_dir / f"summary_{timestamp}.txt"
    generate_summary(all_results, summary_path)

    # Save all results as JSON
    all_results_path = results_dir / f"all_results_{timestamp}.json"
    with open(all_results_path, 'w') as f:
        json.dump([
            {
                'exp_id': r.exp_id,
                'timestamp': r.timestamp,
                'config': r.config,
                'converged': r.converged,
                'final_loss': r.final_loss,
                'translation_error': r.translation_error,
                'rotation_error': r.rotation_error,
                'scale_error': r.scale_error,
                'num_iters': r.num_iters,
                'grid_build_time': r.grid_build_time,
                'registration_time': r.registration_time,
                'success': r.success,
            }
            for r in all_results
        ], f, indent=2)

    print(f"\nAll results saved to: {all_results_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
