#!/usr/bin/env python3
"""
Picotalk training script.

Features:
- AdamW optimizer with cosine LR schedule + warmup
- Gradient accumulation for large effective batch sizes
- Mixed precision training (bf16/fp16)
- Checkpointing and resumption
- Wandb logging
- Evaluation with perplexity
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.config import ModelConfig, get_config
from models.model import build_model
from data_loader import create_dataloaders
from eval.evaluate import DEFAULT_PROMPTS


# Qualitative generation samples, produced alongside each evaluation.
SAMPLE_TOKENS = 64
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 50
SAMPLE_TOKENIZER = 'mistralai/Mistral-7B-v0.1'

# Base RNG seed; each rank uses SEED + rank so ranks draw different batches.
SEED = 1337

# Compile the model with TorchInductor (CUDA only). Costs a minute or two at
# startup, then fuses pointwise ops and cuts kernel-launch overhead.
COMPILE = True


class Trainer:
    """Main training class."""

    def __init__(self, config: Dict[str, Any], model_config: ModelConfig):
        self.config = config
        self.model_config = model_config

        # Setup device
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = config.get('use_amp', True) and self.device == 'cuda'
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Distributed training
        self.is_ddp = config.get('ddp', False)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.on_cuda = str(self.device).startswith('cuda')

        if self.is_ddp:
            # gloo lets the DDP path be smoke-tested on CPU without GPUs
            dist.init_process_group(backend='nccl' if self.on_cuda else 'gloo')
            if self.on_cuda:
                torch.cuda.set_device(self.local_rank)
                self.device = f'cuda:{self.local_rank}'

        # Global rank, not local, so this stays correct across multiple nodes
        self.is_main_process = (not self.is_ddp) or (self.rank == 0)

        # Every rank must draw DIFFERENT data: DDP averages gradients, so if the
        # ranks saw identical batches they would compute identical gradients and
        # the run would do world_size x the work for 1x the data -- silently,
        # with a loss curve that still looks healthy. Offsetting the seed by rank
        # keeps the run reproducible while keeping the ranks' batches distinct.
        torch.manual_seed(SEED + self.rank)

        # Training batches are drawn from a dedicated generator reseeded from
        # (SEED, rank, step) at every optimizer step. That makes the data stream
        # a pure function of the step, so --resume continues the stream instead
        # of replaying it from the beginning. Note this is deliberately NOT done
        # by checkpointing RNG state: only rank 0 writes checkpoints, so
        # restoring its state everywhere would give all ranks identical batches.
        self.data_generator = torch.Generator()

        # Build model
        model = build_model(model_config).to(self.device)

        # Keep the plain module for state_dict/generate: torch.compile prefixes
        # state_dict keys with '_orig_mod.' and DDP with 'module.', neither of
        # which we want baked into checkpoints.
        self.raw_model = model

        if COMPILE and self.on_cuda:
            if self.is_main_process:
                print("Compiling model (first step will be slow)...")
            model = torch.compile(model)

        if self.is_ddp:
            model = DDP(
                model,
                device_ids=[self.local_rank] if self.on_cuda else None,
                # freqs_cis is deterministic and identical on every rank, and
                # there are no other buffers, so skip the per-iteration sync
                broadcast_buffers=False,
            )

        self.model = model

        # Training state
        self.step = 0
        self.epoch = 0
        self.tokens_seen = 0
        self.best_val_loss = float('inf')

        # Setup optimizer and scaler
        self.optimizer = self._build_optimizer()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Learning rate schedule
        self.max_steps = config['max_steps']
        # NOTE: argparse always puts 'warmup_steps' in the config (as None when
        # unset), so .get()'s default never fires -- use `or` to catch None.
        self.warmup_steps = config.get('warmup_steps') or max(100, self.max_steps // 20)

        # Dataloaders
        data_dir = Path(config['data_dir'])
        batch_size = config['batch_size']
        block_size = model_config.block_size

        self.train_dataset, self.val_dataset = create_dataloaders(
            data_dir, batch_size, block_size, self.device
        )

        # Gradient accumulation
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        self.effective_batch_size = batch_size * self.grad_accum_steps * self.world_size

        # Logging
        self.log_interval = config.get('log_interval', 10)
        self.eval_interval = config.get('eval_interval', 500)
        self.save_interval = config.get('save_interval', 1000)

        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Qualitative generation samples, produced alongside each evaluation
        self.sample_prompts = []
        self.sample_tokenizer = None

        if self.is_main_process:
            from transformers import AutoTokenizer
            self.sample_prompts = list(DEFAULT_PROMPTS)
            self.sample_tokenizer = AutoTokenizer.from_pretrained(SAMPLE_TOKENIZER)

        # Wandb
        self.use_wandb = config.get('use_wandb', False) and self.is_main_process
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config.get('wandb_project', 'picotalk'),
                name=config.get('run_name', f"{model_config.n_layer}L-{model_config.n_embd}d"),
                config={**config, **vars(model_config)},
            )

        if self.is_main_process:
            print(f"\n{'='*60}")
            print("Picotalk Training")
            print(f"{'='*60}")
            print(f"Model: {self.raw_model.get_num_params()/1e6:.1f}M parameters")
            print(f"Device: {self.device}")
            print(f"Mixed precision: {self.use_amp} ({self.dtype if self.use_amp else 'fp32'})")
            print(f"Batch size: {batch_size} (effective: {self.effective_batch_size})")
            print(f"Gradient accumulation: {self.grad_accum_steps}")
            tokens_per_step = self.effective_batch_size * block_size
            print(f"Max steps: {self.max_steps:,} (optimizer steps)")
            print(f"Tokens/step: {tokens_per_step:,}")
            print(f"Total token budget: {self.max_steps * tokens_per_step/1e9:.2f}B")
            print(f"Warmup steps: {self.warmup_steps:,}")
            print(f"Learning rate: {config['learning_rate']:.2e}")
            print(f"Weight decay: {config['weight_decay']}")
            if self.is_ddp:
                print(f"Distributed: {self.world_size} GPUs")
            print(f"{'='*60}\n")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        config = self.config

        # Separate parameters into decay/no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases, layer norms, embeddings
            if param.ndim < 2 or 'norm' in name or 'bias' in name or 'embeddings' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': config['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        # The fused CUDA implementation does the whole update in one kernel
        # instead of looping per-parameter tensor
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config['learning_rate'],
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.95)),
            eps=config.get('eps', 1e-8),
            fused=self.on_cuda,
        )

        if self.is_main_process:
            print(f"Optimizer groups: {len(decay_params)} decay, {len(no_decay_params)} no-decay"
                  f" (fused={self.on_cuda})")

        return optimizer

    def get_lr(self) -> float:
        """Cosine learning rate schedule with warmup."""
        lr = self.config['learning_rate']
        min_lr = self.config.get('min_lr', lr * 0.1)

        # Warmup
        if self.step < self.warmup_steps:
            return lr * (self.step + 1) / self.warmup_steps

        # Cosine decay
        if self.step > self.max_steps:
            return min_lr

        decay_ratio = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step."""
        # Forward pass
        with autocast(enabled=self.use_amp, dtype=self.dtype):
            logits, loss = self.model(x, y)
            loss = loss / self.grad_accum_steps  # Scale for accumulation

        # Backward pass
        self.scaler.scale(loss).backward()

        return loss.item() * self.grad_accum_steps  # Unscale for logging

    @torch.no_grad()
    def eval(self) -> float:
        """Evaluate on validation set."""
        if self.val_dataset is None:
            return float('nan')

        self.model.eval()

        eval_steps = self.config.get('eval_steps', 50)
        losses = []

        for _ in range(eval_steps):
            x, y = self.val_dataset.get_batch(
                self.config['batch_size'],
                self.model_config.block_size,
                self.device
            )

            with autocast(enabled=self.use_amp, dtype=self.dtype):
                _, loss = self.model(x, y)

            losses.append(loss.item())

        self.model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample(self) -> list:
        """
        Generate completions for the qualitative prompts.

        Returns a list of (prompt, completion) pairs.
        """
        if not self.sample_prompts:
            return []

        self.model.eval()
        rows = []

        for prompt in self.sample_prompts:
            input_ids = self.sample_tokenizer.encode(
                prompt, return_tensors='pt'
            ).to(self.device)

            with autocast(enabled=self.use_amp, dtype=self.dtype):
                # generate() lives on the raw model, not the DDP wrapper
                output_ids = self.raw_model.generate(
                    input_ids,
                    max_new_tokens=SAMPLE_TOKENS,
                    temperature=SAMPLE_TEMPERATURE,
                    top_k=SAMPLE_TOP_K,
                )

            text = self.sample_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Strip the prompt so the log shows only what the model added
            completion = text[len(prompt):] if text.startswith(prompt) else text
            rows.append((prompt, completion))

        self.model.train()
        return rows

    def log_samples(self):
        """Generate and report qualitative samples (stdout + wandb)."""
        rows = self.sample()
        if not rows:
            return

        print(f"  --- samples @ step {self.step:,} ---")
        for prompt, completion in rows:
            flat = completion.replace('\n', '\\n')
            print(f"    {prompt!r} -> {flat[:160]}")

        if self.use_wandb:
            import wandb
            table = wandb.Table(columns=['step', 'prompt', 'completion'])
            for prompt, completion in rows:
                table.add_data(self.step, prompt, completion)
            wandb.log({'samples': table, 'step': self.step})

    def save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'tokens_seen': self.tokens_seen,
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'model_config': vars(self.model_config),
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.tokens_seen = checkpoint['tokens_seen']
        self.best_val_loss = checkpoint['best_val_loss']

        self.raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.is_main_process:
            print(f"Loaded checkpoint from step {self.step}")

    def train(self):
        """Main training loop."""
        self.model.train()

        start_time = time.time()
        # self.step counts OPTIMIZER steps, so one step consumes a full
        # effective batch: micro-batch x accumulation x ranks.
        tokens_per_step = (
            self.config['batch_size'] * self.grad_accum_steps
            * self.world_size * self.model_config.block_size
        )

        while self.step < self.max_steps:
            # Learning rate is set once per optimizer step, not mid-accumulation
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Reseed per (rank, step) so this step's batches are reproducible
            # and independent of how many steps preceded them in this process.
            self.data_generator.manual_seed(
                SEED * 1_000_003 + self.rank * 100_003 + self.step
            )

            # Accumulate gradients over grad_accum_steps micro-batches
            loss_sum = 0.0
            for micro_step in range(self.grad_accum_steps):
                x, y = self.train_dataset.get_batch(
                    self.config['batch_size'],
                    self.model_config.block_size,
                    self.device,
                    generator=self.data_generator,
                )

                # Under DDP, every backward() would normally all-reduce gradients
                # -- wasteful mid-accumulation, where they are only being summed
                # locally. no_sync() defers the all-reduce to the final
                # micro-batch, cutting communication by grad_accum_steps.
                is_last_micro = (micro_step == self.grad_accum_steps - 1)
                if self.is_ddp and not is_last_micro:
                    with self.model.no_sync():
                        loss_sum += self.train_step(x, y)
                else:
                    loss_sum += self.train_step(x, y)

            loss = loss_sum / self.grad_accum_steps

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.tokens_seen += tokens_per_step
            self.step += 1

            # Logging
            if self.step % self.log_interval == 0 and self.is_main_process:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_per_step * self.log_interval / elapsed

                print(f"Step {self.step:>6,} | Loss {loss:.4f} | LR {lr:.2e} | "
                      f"Tokens/s {tokens_per_sec:>8,.0f} | "
                      f"Total {self.tokens_seen/1e9:.2f}B tokens")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': loss,
                        'train/lr': lr,
                        'train/tokens_per_sec': tokens_per_sec,
                        'train/tokens_seen': self.tokens_seen,
                        'step': self.step,
                    })

                start_time = time.time()

            # Evaluation
            if self.step % self.eval_interval == 0 and self.is_main_process:
                val_loss = self.eval()
                val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')

                print(f"  Validation: Loss {val_loss:.4f} | PPL {val_ppl:.2f}")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'val/loss': val_loss,
                        'val/perplexity': val_ppl,
                        'step': self.step,
                    })

                self.log_samples()

                # Save best model. Only best.pt is rewritten here -- periodic
                # step_N.pt snapshots are the save_interval's job, and writing
                # both on every improving eval doubled checkpoint I/O.
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(self.checkpoint_dir / 'best.pt')
                    print(f"  New best model saved (val_loss={val_loss:.4f})")

            # Checkpointing
            if self.step % self.save_interval == 0 and self.is_main_process:
                checkpoint_path = self.checkpoint_dir / f'step_{self.step}.pt'
                self.save_checkpoint(checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path.name}")

        # Final evaluation and save
        if self.is_main_process:
            print("\nTraining complete!")
            val_loss = self.eval()
            print(f"Final validation loss: {val_loss:.4f}")

            final_path = self.checkpoint_dir / 'final.pt'
            self.save_checkpoint(final_path)
            print(f"Final checkpoint saved: {final_path}")

        if self.is_ddp:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Picotalk")

    # Model
    parser.add_argument('--model', type=str, default='1b', help='Model config name')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data/tokenized',
                        help='Directory with tokenized .bin files')

    # Training
    parser.add_argument('--batch-size', type=int, default=8, help='Per-device batch size')
    parser.add_argument('--grad-accum-steps', type=int, default=16, help='Gradient accumulation')
    parser.add_argument('--max-steps', type=int, default=30000, help='Total training steps')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--min-lr', type=float, default=3e-5, help='Min LR for cosine decay')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup-steps', type=int, default=None, help='Warmup steps (default: max_steps/20)')

    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--ddp', action='store_true', help='Enable distributed training')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10, help='Log every N steps')
    parser.add_argument('--eval-interval', type=int, default=500, help='Eval every N steps')
    parser.add_argument('--eval-steps', type=int, default=50, help='Number of eval steps')
    parser.add_argument('--save-interval', type=int, default=2000, help='Save every N steps')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')

    # Wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='picotalk', help='Wandb project')
    parser.add_argument('--run-name', type=str, default=None, help='Run name')

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Load model config
    model_config = get_config(args.model)

    # Build training config
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'max_steps': args.max_steps,
        'learning_rate': args.learning_rate,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'warmup_steps': args.warmup_steps,
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
        'run_name': args.run_name,
    }

    # Create trainer
    trainer = Trainer(config, model_config)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
