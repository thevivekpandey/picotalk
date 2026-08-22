#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) for Picotalk.

Reuses the pretraining Trainer (train/train.py) with three changes:

1. Data: reads the packed SFT stream from data/prepare_sft_data.py, whose
   loader (train/sft_data_loader.py) sets targets to -1 on non-assistant
   tokens so the loss only covers assistant replies (+ their EOS).
2. Initialization: --init-from loads MODEL WEIGHTS ONLY from a pretraining
   checkpoint; optimizer, LR schedule, and step counter start fresh.
3. Schedule: --epochs derives max_steps from the dataset size, with
   SFT-appropriate defaults (LR 2e-5, weight decay 0.01, short warmup).

Usage:
    # Local smoke test (tiny model, tiny data)
    python3 data/prepare_sft_data.py --datasets openhermes --limit 20
    python3 train/sft_train.py --model test --device cpu --no-amp \
        --init-from '' --batch-size 2 --grad-accum-steps 1 --max-steps 5

    # Real run (4x H100)
    torchrun --nproc_per_node=4 train/sft_train.py --ddp \
        --init-from checkpoints/run_1b/step_45000.pt \
        --epochs 2 --wandb --run-name picotalk-1b-sft

    # Resume an interrupted SFT run (full state, not just weights)
    torchrun --nproc_per_node=4 train/sft_train.py --ddp \
        --resume checkpoints/sft_1b/step_500.pt
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

# train/train.py: imported both as a module (to patch its dataloader factory)
# and for the Trainer class itself.
import train as pretrain
from train import Trainer

from models.config import get_config
from data.chat_format import format_prompt
from sft_data_loader import create_sft_dataloaders

# Chat-style qualitative prompts (the pretraining ones are raw continuations)
SFT_SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "If John has 3 apples and buys 2 more, how many apples does he have?",
    "Explain in one paragraph why the sky is blue.",
]


def _sft_create_dataloaders(data_dir, batch_size, block_size, device):
    """Match the pretraining factory signature; SFTDataset ignores the rest."""
    return create_sft_dataloaders(Path(data_dir))


class SFTTrainer(Trainer):
    """Pretraining Trainer with SFT data and chat-formatted samples."""

    def __init__(self, config, model_config):
        super().__init__(config, model_config)

        # Swap the qualitative prompts for chat-formatted ones so samples
        # exercise the [INST] template the model is being tuned on.
        if self.is_main_process:
            self.sample_prompts = [
                format_prompt([{"role": "user", "content": q}])
                for q in SFT_SAMPLE_QUESTIONS
            ]

    def load_pretrained_weights(self, path: Path):
        """Initialize model weights from a pretraining checkpoint.

        Unlike load_checkpoint(), this leaves the optimizer, scaler, LR
        schedule, and step counter fresh -- SFT is a new run, not a resume.
        """
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.raw_model.load_state_dict(state_dict)

        if self.is_main_process:
            src_step = checkpoint.get('step', '?')
            src_tokens = checkpoint.get('tokens_seen', 0)
            print(f"Initialized weights from {path} "
                  f"(pretrain step {src_step}, {src_tokens/1e9:.1f}B tokens)")


def main():
    parser = argparse.ArgumentParser(description="SFT for Picotalk")

    # Model
    parser.add_argument('--model', type=str, default='1b', help='Model config name')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data/sft_tokenized',
                        help='Directory with sft_*_tokens.bin / sft_*_mask.bin')

    # Initialization
    parser.add_argument('--init-from', type=str,
                        default='./checkpoints/run_1b/step_45000.pt',
                        help='Pretraining checkpoint for initial weights '
                             '(pass an empty string to start from scratch)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume an interrupted SFT run (overrides --init-from)')

    # Training
    parser.add_argument('--epochs', type=float, default=2.0,
                        help='Passes over the SFT data (sets max_steps)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Explicit step count (overrides --epochs)')
    parser.add_argument('--batch-size', type=int, default=8, help='Per-device batch size')
    parser.add_argument('--grad-accum-steps', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Peak learning rate')
    parser.add_argument('--min-lr', type=float, default=2e-6, help='Min LR for cosine decay')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Warmup steps (default: max_steps/20, min 10)')

    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--ddp', action='store_true', help='Enable distributed training')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N steps')
    parser.add_argument('--eval-interval', type=int, default=100, help='Eval every N steps')
    parser.add_argument('--eval-steps', type=int, default=20, help='Number of eval batches')
    parser.add_argument('--save-interval', type=int, default=500, help='Save every N steps')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/sft_1b',
                        help='Checkpoint directory')

    # Wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='picotalk', help='Wandb project')
    parser.add_argument('--run-name', type=str, default=None, help='Run name')

    args = parser.parse_args()

    model_config = get_config(args.model)

    # Derive max_steps from --epochs over the packed token stream.
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        tokens_file = Path(args.data_dir) / 'sft_train_tokens.bin'
        if not tokens_file.exists():
            parser.error(f"{tokens_file} not found -- run data/prepare_sft_data.py first")
        train_tokens = tokens_file.stat().st_size // 2  # uint16 = 2 bytes/token

        world_size = int(os.environ.get('WORLD_SIZE', 1))
        tokens_per_step = (args.batch_size * args.grad_accum_steps
                           * world_size * model_config.block_size)
        max_steps = max(1, math.ceil(args.epochs * train_tokens / tokens_per_step))
        print(f"SFT schedule: {train_tokens:,} train tokens x {args.epochs} epochs"
              f" / {tokens_per_step:,} tokens/step = {max_steps:,} steps")

    # Trainer treats a None/0 warmup as max(100, max_steps/20); SFT runs are
    # short, so compute an explicit smaller warmup here instead.
    warmup_steps = args.warmup_steps or max(10, max_steps // 20)

    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'max_steps': max_steps,
        'learning_rate': args.learning_rate,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'warmup_steps': warmup_steps,
        'device': args.device,
        'use_amp': not args.no_amp,
        'ddp': args.ddp,
        'log_interval': args.log_interval,
        'eval_interval': args.eval_interval,
        'eval_steps': args.eval_steps,
        'save_interval': args.save_interval,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.wandb,
        'wandb_project': args.wandb_project,
        'run_name': args.run_name or 'picotalk-sft',
        'init_from': args.init_from,
    }

    # Route the Trainer's dataloader factory to the mask-aware SFT loader.
    pretrain.create_dataloaders = _sft_create_dataloaders

    trainer = SFTTrainer(config, model_config)

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    elif args.init_from:
        trainer.load_pretrained_weights(Path(args.init_from))
    else:
        print("WARNING: no --init-from checkpoint; SFT will start from random weights")

    trainer.train()


if __name__ == '__main__':
    main()
