#!/usr/bin/env python
"""Quick test of a few key parameter combinations."""

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

import taichi as ti
ti.init(arch=ti.cuda, log_level=ti.ERROR)
device = 'cuda'

from gmm_point_alignment.unified_registration import (
    UnifiedRegistration, UnifiedConfig, RegistrationMethod
)
from gmm_point_alignment.transform_utils import se3_exp
from misc.hier_IO import load_hier_to_torch


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup
    hier_path = Path('../data/merged.hier')
    hier_scene = load_hier_to_torch(hier_path, device=device)
    scene = hier_scene.gaussian_scene

    np.random.seed(42)
    sample_indices = torch.randperm(len(scene.position))[:1000]
    pointcloud = scene.position[sample_indices].clone()

    xi_true = torch.tensor([0.3, -0.2, 0.1, 0.05, -0.03, 0.02], device=device)
    T_true = se3_exp(xi_true)
    pointcloud_transformed = (T_true[:3, :3] @ pointcloud.T).T + T_true[:3, 3]
    T_true_inv = torch.inverse(T_true)

    # Test configs
    test_configs = [
        {"name": "baseline", "lr_t": 0.01, "lr_r": 0.01},
        {"name": "small_t_large_r", "lr_t": 0.001, "lr_r": 0.05},
        {"name": "very_small_t", "lr_t": 0.0001, "lr_r": 0.05},
        {"name": "balanced", "lr_t": 0.005, "lr_r": 0.02},
    ]

    results = []

    for cfg in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {cfg['name']}")
        print(f"  lr_translation={cfg['lr_t']}, lr_rotation={cfg['lr_r']}")

        config = UnifiedConfig(
            method=RegistrationMethod.MLE,
            mle_num_iters=100,
            mle_lr_translation=cfg['lr_t'],
            mle_lr_rotation=cfg['lr_r'],
            mle_multi_init=True,
            mle_num_init=3,
        )

        reg = UnifiedRegistration(config)
        result = reg.register(scene, pointcloud_transformed)

        # Compute errors
        t_error = (result.t - T_true_inv[:3, 3]).norm().item()
        R_rel = result.R.T @ T_true_inv[:3, :3]
        angle_error = torch.acos(torch.clamp((R_rel.trace() - 1) / 2, -1, 1)).item() * 180 / np.pi

        print(f"  Results:")
        print(f"    Loss: {result.error:.4f}")
        print(f"    T-error: {t_error:.4f}m")
        print(f"    R-error: {angle_error:.2f}°")
        print(f"    Converged: {result.converged}")

        results.append({
            'name': cfg['name'],
            'lr_translation': cfg['lr_t'],
            'lr_rotation': cfg['lr_r'],
            'loss': result.error,
            'translation_error': t_error,
            'rotation_error': angle_error,
            'converged': result.converged,
            'num_iters': result.num_iters,
        })

    # Save results
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"quick_test_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary table
    print("\nSummary:")
    print(f"{'Config':<20} {'lr_t':<8} {'lr_r':<8} {'T-err(m)':<10} {'R-err(°)':<10} {'Loss':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['lr_translation']:<8.4f} {r['lr_rotation']:<8.4f} "
              f"{r['translation_error']:<10.4f} {r['rotation_error']:<10.2f} {r['loss']:<8.2f}")


if __name__ == "__main__":
    main()
