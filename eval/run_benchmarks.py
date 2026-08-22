#!/usr/bin/env python3
"""
Run benchmark evaluations on Picotalk models using LM Evaluation Harness.

Usage:
    # Evaluate on all standard benchmarks
    python eval/run_benchmarks.py --checkpoint checkpoints/run_1b/step_45000.pt

    # Evaluate on specific benchmarks
    python eval/run_benchmarks.py --checkpoint checkpoints/run_1b/step_45000.pt --tasks hellaswag,mmlu

    # Quick test (small subset)
    python eval/run_benchmarks.py --checkpoint checkpoints/run_1b/step_45000.pt --limit 10
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from eval.lm_eval_wrapper import load_picotalk_model


def run_evaluation(
    checkpoint_path: str,
    tasks: str = "default",
    output_dir: str = "eval/results",
    limit: int = None,
    device: str = "cuda",
    num_fewshot: int = None,
):
    """
    Run evaluation on specified tasks.

    Args:
        checkpoint_path: Path to model checkpoint
        tasks: Comma-separated list of tasks or "default"/"all"
        output_dir: Directory to save results
        limit: Limit number of examples per task (for testing)
        device: Device to use
        num_fewshot: Number of few-shot examples (None = task default)
    """
    # Import lm_eval here to avoid issues if not installed
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("ERROR: lm-eval not installed. Run: pip install lm-eval")
        sys.exit(1)

    # Define task suites
    task_suites = {
        "default": [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "piqa",
        ],
        "full": [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "piqa",
            "mmlu",
            "truthfulqa_mc2",
        ],
        "reasoning": [
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "piqa",
        ],
        "knowledge": [
            "mmlu",
            "truthfulqa_mc2",
        ],
        "code": [
            "humaneval",
        ],
        "math": [
            "gsm8k",
        ],
    }

    # Get task list
    if tasks in task_suites:
        task_list = task_suites[tasks]
        print(f"\nUsing '{tasks}' task suite: {task_list}")
    else:
        task_list = tasks.split(",")
        print(f"\nUsing custom tasks: {task_list}")

    # Load model
    print(f"\n{'='*80}")
    print(f"Loading Picotalk model from: {checkpoint_path}")
    print(f"{'='*80}\n")

    # We need to create a custom HFLM-compatible wrapper
    # For now, let's use a simpler approach with direct API

    print(f"\n{'='*80}")
    print(f"Running evaluation on {len(task_list)} tasks")
    print(f"{'='*80}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get checkpoint name for results file
    ckpt_name = Path(checkpoint_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"{ckpt_name}_{timestamp}.json"

    # Run evaluation using lm-eval CLI approach
    import subprocess

    # Build command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path},dtype=bfloat16",
        "--tasks", ",".join(task_list),
        "--device", device,
        "--output_path", str(output_path),
        "--log_samples",
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    print(f"Command: {' '.join(cmd)}\n")
    print("Note: This may take a while depending on the number of tasks...\n")

    # Actually, lm-eval expects HuggingFace format. Let me create a simpler direct implementation
    print("ERROR: Direct lm-eval integration requires HuggingFace format.")
    print("\nTo evaluate Picotalk, you have two options:")
    print("\n1. Use the direct evaluation script (recommended for now):")
    print("   python eval/run_benchmarks_direct.py --checkpoint <path>")
    print("\n2. Export model to HuggingFace format first:")
    print("   python export_to_hf.py --checkpoint <path>")
    print("   lm_eval --model hf --model_args pretrained=<hf_path> --tasks <tasks>")

    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Picotalk on benchmarks")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="default",
        help="Tasks to evaluate: 'default', 'full', 'reasoning', 'knowledge', 'code', 'math', or comma-separated list",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: task-specific)",
    )

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        tasks=args.tasks,
        output_dir=args.output_dir,
        limit=args.limit,
        device=args.device,
        num_fewshot=args.num_fewshot,
    )


if __name__ == "__main__":
    main()
