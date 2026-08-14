#!/usr/bin/env python3
"""
Scaling law experiments for Picotalk.

Run small-scale experiments to empirically derive scaling laws before committing
to the final 1B parameter training run.

Experiment matrix:
- Model sizes: 50M, 100M, 200M, 350M
- Data sizes: 1B, 3B, 5B, 10B tokens
- Goal: Find optimal (N, D) allocation for compute budget
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List


# Scaling experiment configurations
SCALING_CONFIGS = {
    # Small models for quick experiments
    "50m_1b": {
        "model": "50m",
        "max_steps": 1000,
        "tokens": 1e9,
        "batch_size": 16,
        "grad_accum": 8,
    },
    "50m_3b": {
        "model": "50m",
        "max_steps": 3000,
        "tokens": 3e9,
        "batch_size": 16,
        "grad_accum": 8,
    },

    # 100M experiments
    "100m_1b": {
        "model": "100m",
        "max_steps": 1000,
        "tokens": 1e9,
        "batch_size": 12,
        "grad_accum": 10,
    },
    "100m_3b": {
        "model": "100m",
        "max_steps": 3000,
        "tokens": 3e9,
        "batch_size": 12,
        "grad_accum": 10,
    },
    "100m_5b": {
        "model": "100m",
        "max_steps": 5000,
        "tokens": 5e9,
        "batch_size": 12,
        "grad_accum": 10,
    },

    # 200M experiments
    "200m_3b": {
        "model": "200m",
        "max_steps": 3000,
        "tokens": 3e9,
        "batch_size": 8,
        "grad_accum": 16,
    },
    "200m_5b": {
        "model": "200m",
        "max_steps": 5000,
        "tokens": 5e9,
        "batch_size": 8,
        "grad_accum": 16,
    },
    "200m_10b": {
        "model": "200m",
        "max_steps": 10000,
        "tokens": 10e9,
        "batch_size": 8,
        "grad_accum": 16,
    },

    # 350M experiments
    "350m_5b": {
        "model": "350m",
        "max_steps": 5000,
        "tokens": 5e9,
        "batch_size": 6,
        "grad_accum": 20,
    },
    "350m_10b": {
        "model": "350m",
        "max_steps": 10000,
        "tokens": 10e9,
        "batch_size": 6,
        "grad_accum": 20,
    },
}


def compute_steps_for_tokens(
    target_tokens: float,
    batch_size: int,
    grad_accum: int,
    block_size: int = 2048,
    num_gpus: int = 1,
) -> int:
    """Compute number of training steps needed to see target_tokens."""
    tokens_per_step = batch_size * grad_accum * block_size * num_gpus
    return int(target_tokens / tokens_per_step)


def run_experiment(
    name: str,
    config: Dict,
    data_dir: str = "./data/tokenized",
    checkpoint_dir: str = "./checkpoints/scaling",
    wandb_project: str = "picotalk-scaling",
    device: str = "cuda",
    dry_run: bool = False,
):
    """Run a single scaling experiment."""
    # Compute actual steps needed
    max_steps = compute_steps_for_tokens(
        config['tokens'],
        config['batch_size'],
        config['grad_accum'],
    )

    # Build command
    cmd = [
        "python3", "train/train.py",
        "--model", config['model'],
        "--data-dir", data_dir,
        "--batch-size", str(config['batch_size']),
        "--grad-accum-steps", str(config['grad_accum']),
        "--max-steps", str(max_steps),
        "--learning-rate", "3e-4",
        "--min-lr", "3e-5",
        "--weight-decay", "0.1",
        "--grad-clip", "1.0",
        "--warmup-steps", str(max(100, max_steps // 20)),
        "--eval-interval", "500",
        "--save-interval", "2000",
        "--checkpoint-dir", f"{checkpoint_dir}/{name}",
        "--wandb",
        "--wandb-project", wandb_project,
        "--run-name", name,
        "--device", device,
    ]

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    print(f"Model: {config['model']}")
    print(f"Target tokens: {config['tokens']/1e9:.1f}B")
    print(f"Steps: {max_steps:,}")
    print(f"Batch size: {config['batch_size']} x {config['grad_accum']} = {config['batch_size'] * config['grad_accum']}")
    print(f"{'='*60}")

    if dry_run:
        print("\nDry run - command:")
        print(" ".join(cmd))
        return

    # Run training
    print("\nStarting training...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ Experiment {name} failed!")
        return False

    print(f"✅ Experiment {name} complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run scaling law experiments")

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="List of experiments to run (default: all)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/tokenized",
        help="Tokenized data directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/scaling",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="picotalk-scaling",
        help="Wandb project name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running"
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\nAvailable scaling experiments:\n")
        print(f"{'Name':<15} {'Model':<8} {'Tokens':<10} {'Steps':<8} {'Batch':<12}")
        print("-" * 60)
        for name, cfg in SCALING_CONFIGS.items():
            steps = compute_steps_for_tokens(cfg['tokens'], cfg['batch_size'], cfg['grad_accum'])
            batch_str = f"{cfg['batch_size']}x{cfg['grad_accum']}"
            print(f"{name:<15} {cfg['model']:<8} {cfg['tokens']/1e9:>6.1f}B   {steps:>7,}  {batch_str:<12}")
        return

    # Select experiments to run
    if args.experiments:
        experiments = {k: v for k, v in SCALING_CONFIGS.items() if k in args.experiments}
        if not experiments:
            print(f"❌ No experiments found matching: {args.experiments}")
            print(f"Available: {list(SCALING_CONFIGS.keys())}")
            return
    else:
        experiments = SCALING_CONFIGS

    # Run experiments
    results = {}
    for name, config in experiments.items():
        success = run_experiment(
            name,
            config,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            wandb_project=args.wandb_project,
            device=args.device,
            dry_run=args.dry_run,
        )
        results[name] = success

    # Summary
    if not args.dry_run:
        print(f"\n{'='*60}")
        print("Scaling Experiments Summary")
        print(f"{'='*60}")
        for name, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {name}")


if __name__ == "__main__":
    main()
