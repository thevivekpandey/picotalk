#!/usr/bin/env python3
"""
GRPO on GSM8K for Picotalk, single GPU.

Two copies of the model share the GPU:

  * the TRAINING copy: the native Picotalk Transformer in fp32 (autocast bf16
    for the matmuls), updated with AdamW. Checkpoints are written in the same
    format as train.py / sft_train.py, so every tool under eval/ reads them.
  * the ROLLOUT copy: the same weights inside vLLM (bf16, paged KV cache), which
    generates the samples. After every optimizer step the training weights are
    converted to the HuggingFace Llama layout (eval/export_hf.py) and pushed
    into the vLLM worker in place, so rollouts always come from the current
    policy.

Algorithm (one optimizer step per batch of rollouts, so this is pure on-policy
and the PPO ratio is identically 1):

    for each of B prompts: sample G completions at temperature 1
    reward  = 1.0 if the answer after "####" (last number as fallback) matches
            + 0.1 if the completion contains "####"
            = 0.0 if the completion hit max_tokens without finishing
    advantage = (reward - group mean) / (group std)      per prompt group
    groups with zero std carry no signal and are dropped from the loss
    loss    = - sum_over_completion_tokens(advantage * log p(token)) / total_tokens

No KL term and no reference model: for verifiable rewards the low learning rate
and gradient clipping are enough to keep the policy from running away.

Prompt pool: GSM8K train, minus a held-out validation slice that is scored
greedily every --eval-interval steps. With --min-solved N the pool is limited to
problems that eval/gsm8k_passk.py solved at least N times out of 32, which
keeps most groups informative while pass@1 is still low. Widen to the full
set later by resuming with --min-solved 0.

Usage:
    # Smoke test (small model, few steps)
    python3 train/grpo_train.py --init-from checkpoints/sft_1b/final.pt \
        --prompts-per-step 4 --group-size 4 --max-steps 2 --eval-size 8

    # Real run
    python3 train/grpo_train.py --init-from checkpoints/sft_1b/final.pt \
        --min-solved 1 --wandb --run-name picotalk-1b-grpo

    # Resume (full state) and widen to every training problem
    python3 train/grpo_train.py --resume checkpoints/grpo_1b/step_200.pt --min-solved 0
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

# Must be set before vllm is imported: the engine runs inside this process so
# that the training weights can be handed to it as tensors.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "eval"))

from models.config import ModelConfig
from models.model import Transformer
import export_hf
import gsm8k_passk as grader  # extract_answer / is_correct / gold_answer / hf_export_dir

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
PAD_ID = 0  # <unk>; masked out of the loss, invisible to earlier positions


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_problems(args):
    """Returns (train_pool, val_set); each a list of (idx, question, gold)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = []
    for i, ex in enumerate(ds):
        gold = grader.gold_answer(ex["answer"])
        if gold:
            problems.append((i, ex["question"], gold))

    rng = random.Random(args.seed)
    order = list(range(len(problems)))
    rng.shuffle(order)
    val_idx = set(order[: args.eval_size])
    val = [problems[i] for i in sorted(val_idx)]
    pool = [p for i, p in enumerate(problems) if i not in val_idx]

    if args.min_solved > 0:
        path = Path(args.passk_jsonl)
        if not path.exists():
            sys.exit(f"--min-solved {args.min_solved} needs {path} (run eval/gsm8k_passk.py)")
        solved = {}
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                solved[row["idx"]] = row["n_correct"]
        before = len(pool)
        pool = [p for p in pool if solved.get(p[0], 0) >= args.min_solved]
        print(f"Prompt pool: {len(pool)} of {before} train problems solved >= {args.min_solved} "
              f"times in {path.name}")
    else:
        print(f"Prompt pool: all {len(pool)} train problems")
    print(f"Validation: {len(val)} held-out train problems")
    return pool, val


def encode_prompt(tokenizer, question: str) -> list:
    """[BOS] [INST] question [/INST] -- the exact shape SFT and the eval use."""
    return [tokenizer.bos_token_id] + tokenizer.encode(
        f"[INST] {question.strip()} [/INST]", add_special_tokens=False
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def score_completion(text: str, gold: str, truncated: bool, format_bonus: float):
    """Returns (reward, correct, uses_hash)."""
    uses_hash = "####" in text
    correct = grader.is_correct(grader.extract_answer(text), gold)
    if truncated:
        return 0.0, correct, uses_hash
    return float(correct) + (format_bonus if uses_hash else 0.0), correct, uses_hash


# ---------------------------------------------------------------------------
# vLLM weight sync
# ---------------------------------------------------------------------------

def _worker_load_weights(worker, weights):
    """Runs inside the vLLM worker (same process): copy weights in place."""
    return worker.get_model().load_weights(weights)


def sync_weights(llm, model: Transformer, cfg: ModelConfig) -> int:
    """Push the training weights into vLLM. Returns the number of tensors loaded."""
    hf = export_hf.to_hf_state_dict(cfg, model.state_dict())
    weights = [(name, w.detach().to(torch.bfloat16)) for name, w in hf.items()]
    loaded = llm.collective_rpc(_worker_load_weights, args=(weights,))[0]
    return len(loaded)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda"
        self.step = 0
        self.best_val_acc = -1.0

        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.eos_id = self.tokenizer.eos_token_id

        # --- training copy -------------------------------------------------
        src = Path(args.resume or args.init_from)
        ckpt = torch.load(src, map_location="cpu")
        cfg_dict = ckpt["model_config"]
        valid = set(ModelConfig.__dataclass_fields__)
        self.model_config = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
        self.model = Transformer(self.model_config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        # fp32 parameters + autocast, never .to(dtype): that would destroy the
        # complex RoPE buffer (see models/model.py).
        self.model = self.model.to(self.device)
        self.model.train()
        print(f"Training copy: {self.model.get_num_params() / 1e6:.1f}M params from {src} "
              f"(step {ckpt.get('step', '?')})")

        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (no_decay if p.ndim < 2 or "norm" in name or "embeddings" in name else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-8, fused=True,
        )

        if args.resume:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.step = ckpt["step"]
            self.best_val_acc = ckpt.get("best_val_acc", -1.0)
            print(f"Resumed optimizer state at step {self.step}")
        del ckpt

        # --- rollout copy ----------------------------------------------------
        # vLLM loads weights from an HF export on disk; the export of the
        # starting checkpoint is reused if present. Whatever it holds, the
        # training weights are pushed in right after, so the two always match.
        from vllm import LLM
        hf_dir = grader.hf_export_dir(Path(args.init_from))
        self.llm = LLM(
            model=str(hf_dir),
            dtype="bfloat16",
            max_model_len=self.model_config.block_size,
            gpu_memory_utilization=args.vllm_gpu_memory,
            seed=args.seed,
        )
        n = sync_weights(self.llm, self.model, self.model_config)
        print(f"Rollout copy: vLLM on {hf_dir.name}, {n} tensors synced from the training copy")

        # --- data ------------------------------------------------------------
        self.pool, self.val = load_problems(args)

        self.use_wandb = args.wandb
        if self.use_wandb:
            import wandb
            wandb.init(project=args.wandb_project, name=args.run_name,
                       config={**vars(args), **vars(self.model_config)})

        self.rollout_log = open(self.checkpoint_dir / "rollouts.jsonl", "a")

    # ----------------------------------------------------------------------
    def lr_at(self, step: int) -> float:
        if step < self.args.warmup_steps:
            return self.args.learning_rate * (step + 1) / self.args.warmup_steps
        return self.args.learning_rate

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, batch):
        """Sample G completions per prompt. Returns a list of per-prompt groups."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        a = self.args
        prompts = [encode_prompt(self.tokenizer, q) for _, q, _ in batch]
        params = SamplingParams(
            n=a.group_size, temperature=a.temperature, top_p=a.top_p,
            max_tokens=a.max_new_tokens, seed=a.seed * 1_000_003 + self.step,
        )
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
        )

        groups = []
        for (idx, question, gold), prompt_ids, out in zip(batch, prompts, outputs):
            samples = []
            for o in out.outputs:
                ids = list(o.token_ids)
                truncated = o.finish_reason == "length"
                # Train on the EOS that ended the turn: stopping is part of the
                # behaviour being rewarded. vLLM may or may not include it.
                if not truncated and (not ids or ids[-1] != self.eos_id):
                    ids.append(self.eos_id)
                text_ids = ids[:-1] if ids and ids[-1] == self.eos_id else ids
                text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
                reward, correct, uses_hash = score_completion(text, gold, truncated, a.format_bonus)
                samples.append({
                    "ids": ids, "text": text, "reward": reward,
                    "correct": correct, "uses_hash": uses_hash, "truncated": truncated,
                })
            groups.append({"idx": idx, "question": question, "gold": gold,
                           "prompt_ids": prompt_ids, "samples": samples})
        return groups

    # ----------------------------------------------------------------------
    def build_examples(self, groups):
        """Group-normalised advantages; drops zero-variance groups."""
        examples = []
        n_informative = 0
        for g in groups:
            rewards = [s["reward"] for s in g["samples"]]
            mean = statistics.fmean(rewards)
            std = statistics.pstdev(rewards)
            if std < 1e-6:
                continue
            n_informative += 1
            for s in g["samples"]:
                adv = (s["reward"] - mean) / (std if self.args.std_normalize else 1.0)
                examples.append((g["prompt_ids"], s["ids"], adv))
        return examples, n_informative

    # ----------------------------------------------------------------------
    def train_on(self, examples) -> dict:
        """One optimizer step over all examples, token-level normalised loss."""
        a = self.args
        total_tokens = sum(len(c) for _, c, _ in examples)
        if total_tokens == 0:
            return {"loss": float("nan"), "grad_norm": float("nan")}

        lr = self.lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        # Length-sorted micro-batches keep padding small.
        examples = sorted(examples, key=lambda e: len(e[0]) + len(e[1]))
        loss_sum = 0.0
        for start in range(0, len(examples), a.micro_batch_size):
            micro = examples[start:start + a.micro_batch_size]
            width = max(len(p) + len(c) for p, c, _ in micro)
            tokens = torch.full((len(micro), width), PAD_ID, dtype=torch.long)
            weight = torch.zeros((len(micro), width), dtype=torch.float32)
            for i, (p, c, adv) in enumerate(micro):
                seq = p + c
                tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                # weight[t] applies to predicting tokens[t]; completion tokens only
                weight[i, len(p): len(seq)] = adv
            tokens = tokens.to(self.device, non_blocking=True)
            weight = weight.to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = self.model(tokens)
            # log p(tokens[t] | tokens[<t]) from logits at t-1
            logp = -F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                reduction="none",
            ).view(len(micro), width - 1)
            loss = -(weight[:, 1:] * logp).sum() / total_tokens
            loss.backward()
            loss_sum += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_sum, "grad_norm": float(grad_norm), "lr": lr}

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> float:
        """Greedy accuracy on the held-out validation problems."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=encode_prompt(self.tokenizer, q))
                   for _, q, _ in self.val]
        params = SamplingParams(temperature=0.0, max_tokens=self.args.max_new_tokens)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        correct = 0
        for (_, _, gold), out in zip(self.val, outputs):
            text = self.tokenizer.decode(out.outputs[0].token_ids, skip_special_tokens=True)
            correct += grader.is_correct(grader.extract_answer(text), gold)
        return 100.0 * correct / len(self.val)

    # ----------------------------------------------------------------------
    def save(self, path: Path):
        torch.save({
            "step": self.step,
            "epoch": 0,
            "tokens_seen": 0,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {**vars(self.args), "trainer": "grpo"},
            "model_config": vars(self.model_config),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": float("nan"),
        }, path)

    def log_rollouts(self, groups, n_groups: int = 2):
        for g in groups[:n_groups]:
            self.rollout_log.write(json.dumps({
                "step": self.step, "idx": g["idx"], "question": g["question"], "gold": g["gold"],
                "samples": [{k: s[k] for k in ("text", "reward", "correct", "uses_hash", "truncated")}
                            for s in g["samples"]],
            }) + "\n")
        self.rollout_log.flush()

    # ----------------------------------------------------------------------
    def train(self):
        a = self.args
        print(f"\n{'=' * 60}\nPicotalk GRPO\n{'=' * 60}")
        print(f"Prompts/step: {a.prompts_per_step}  Group size: {a.group_size}  "
              f"Completions/step: {a.prompts_per_step * a.group_size}")
        print(f"LR: {a.learning_rate:.1e}  warmup {a.warmup_steps}  grad clip {a.grad_clip}  "
              f"micro-batch {a.micro_batch_size}")
        print(f"Rollouts: T={a.temperature} top_p={a.top_p} max {a.max_new_tokens} tokens  "
              f"format bonus {a.format_bonus}  std-normalise {a.std_normalize}")
        print(f"Steps: {self.step} -> {a.max_steps}\n{'=' * 60}\n")

        if self.step == 0:
            val_acc = self.evaluate()
            self.best_val_acc = val_acc
            print(f"  Validation (greedy) @ step 0: {val_acc:.2f}%")
            if self.use_wandb:
                import wandb
                wandb.log({"val/accuracy": val_acc}, step=0)

        while self.step < a.max_steps:
            t0 = time.time()
            rng = random.Random(a.seed * 7_919 + self.step)
            batch = rng.sample(self.pool, min(a.prompts_per_step, len(self.pool)))
            groups = self.rollout(batch)
            t_gen = time.time() - t0

            examples, n_informative = self.build_examples(groups)
            stats = self.train_on(examples)
            t_train = time.time() - t0 - t_gen

            n_synced = sync_weights(self.llm, self.model, self.model_config)
            t_sync = time.time() - t0 - t_gen - t_train
            self.step += 1

            flat = [s for g in groups for s in g["samples"]]
            n = len(flat)
            metrics = {
                "train/reward": statistics.fmean(s["reward"] for s in flat),
                "train/correct": 100 * sum(s["correct"] for s in flat) / n,
                "train/uses_hash": 100 * sum(s["uses_hash"] for s in flat) / n,
                "train/truncated": 100 * sum(s["truncated"] for s in flat) / n,
                "train/solved_prompts": 100 * sum(any(s["correct"] for s in g["samples"]) for g in groups) / len(groups),
                "train/informative_groups": 100 * n_informative / len(groups),
                "train/completion_tokens": statistics.fmean(len(s["ids"]) for s in flat),
                "train/loss": stats["loss"],
                "train/grad_norm": stats["grad_norm"],
                "train/lr": stats.get("lr", 0.0),
                "time/generate_s": t_gen, "time/train_s": t_train, "time/sync_s": t_sync,
            }
            print(f"Step {self.step:>5} | reward {metrics['train/reward']:.3f} | "
                  f"correct {metrics['train/correct']:5.1f}% | #### {metrics['train/uses_hash']:5.1f}% | "
                  f"trunc {metrics['train/truncated']:4.1f}% | signal {metrics['train/informative_groups']:5.1f}% | "
                  f"len {metrics['train/completion_tokens']:5.0f} | gnorm {stats['grad_norm']:.2f} | "
                  f"gen {t_gen:.0f}s train {t_train:.0f}s sync {t_sync:.1f}s", flush=True)

            if self.step % a.log_interval == 0:
                self.log_rollouts(groups)

            if self.step % a.eval_interval == 0 or self.step == a.max_steps:
                val_acc = self.evaluate()
                metrics["val/accuracy"] = val_acc
                print(f"  Validation (greedy): {val_acc:.2f}%  (best {max(val_acc, self.best_val_acc):.2f}%)")
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save(self.checkpoint_dir / "best.pt")
                    print("  New best saved")

            if self.use_wandb:
                import wandb
                wandb.log(metrics, step=self.step)

            if self.step % a.save_interval == 0:
                path = self.checkpoint_dir / f"step_{self.step}.pt"
                self.save(path)
                print(f"  Checkpoint saved: {path.name}")

        self.save(self.checkpoint_dir / "final.pt")
        print(f"\nDone. Final checkpoint: {self.checkpoint_dir / 'final.pt'}  "
              f"best val {self.best_val_acc:.2f}%")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="GRPO on GSM8K for Picotalk (single GPU)")
    # Model
    p.add_argument("--init-from", default="checkpoints/sft_1b/final.pt",
                   help="starting policy (weights only)")
    p.add_argument("--resume", default=None, help="GRPO checkpoint to resume (full state)")
    p.add_argument("--checkpoint-dir", default="checkpoints/grpo_1b")
    # Prompts
    p.add_argument("--passk-jsonl", default="eval/results/gsm8k_passk/sft_1b_final_train_n32.jsonl")
    p.add_argument("--min-solved", type=int, default=0,
                   help="keep train problems solved >= N/32 times in --passk-jsonl; 0 = all")
    p.add_argument("--eval-size", type=int, default=256, help="held-out train problems for validation")
    # Rollouts
    p.add_argument("--prompts-per-step", type=int, default=64)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--format-bonus", type=float, default=0.1)
    p.add_argument("--no-std-normalize", dest="std_normalize", action="store_false",
                   help="mean-centre advantages only (Dr. GRPO style)")
    p.add_argument("--vllm-gpu-memory", type=float, default=0.4,
                   help="fraction of the GPU vLLM may use; the rest is the trainer's")
    # Optimisation
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2500)
    # Logging / checkpoints
    p.add_argument("--log-interval", type=int, default=10, help="rollout dump cadence")
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="picotalk")
    p.add_argument("--run-name", default="picotalk-1b-grpo")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    GRPOTrainer(args).train()


if __name__ == "__main__":
    main()
