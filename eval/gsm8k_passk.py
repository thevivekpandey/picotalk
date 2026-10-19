#!/usr/bin/env python3
"""
pass@k on GSM8K for a Picotalk SFT checkpoint.

Samples N solutions per problem under vLLM and grades each one, so the
per-problem solve counts tell us how much signal there is for rejection
sampling / RL before any of it is built:

    solved 0/N   -- nothing to learn from yet
    solved 1..   -- the band where RFT and GRPO do their work
    solved most  -- already easy, contributes little

Prompts are built exactly as the benchmark runner does in --chat-template mode
([BOS] + "[INST] {question} [/INST]"), so pass@1 here at low temperature would
match the benchmark's GSM8K number.

Grading prefers the number after "####" (the GSM8K convention the SFT data
taught) and falls back to the last number in the text. The benchmark's
legacy last-number verdict is recorded alongside so the grader gap is visible.

Usage:
    # Full train split, 32 samples each (default)
    python3 eval/gsm8k_passk.py --checkpoint checkpoints/sft_1b/final.pt

    # Smoke test
    python3 eval/gsm8k_passk.py --checkpoint checkpoints/sft_1b/final.pt --limit 8 --n 4

    # Re-summarise an existing run without generating anything
    python3 eval/gsm8k_passk.py --summarize eval/results/gsm8k_passk/sft_1b_final_train_n32.jsonl

Output (under --output-dir):
    {tag}.jsonl         one line per problem: question, gold, every sample + verdict
    {tag}_summary.json  pass@k table, difficulty buckets, grader comparison

The .jsonl is appended as batches finish and re-running resumes from it, so an
interrupted run keeps what it has.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
NUMBER_RE = r"-?\d+(?:,\d{3})*(?:\.\d+)?"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def gold_answer(solution: str) -> str:
    """The number after '####' in a GSM8K reference solution."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", solution)
    return match.group(1).replace(",", "") if match else ""


def extract_answer(text: str) -> str:
    """Number after the last '####' if present, else the last number in the text."""
    hashed = re.findall(r"####\s*(" + NUMBER_RE + ")", text)
    if hashed:
        return hashed[-1].replace(",", "")
    return extract_last_number(text)


def extract_last_number(text: str) -> str:
    """The benchmark runner's current heuristic, kept for comparison."""
    numbers = re.findall(NUMBER_RE, text)
    return numbers[-1].replace(",", "") if numbers else ""


def is_correct(pred: str, gold: str) -> bool:
    if not pred:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred == gold


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from Chen et al. 2021: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(rows: list, n: int) -> dict:
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= n]
    counts = [r["n_correct"] for r in rows]
    counts_legacy = [r["n_correct_legacy"] for r in rows]
    total = len(rows)

    table = {f"pass@{k}": 100 * sum(pass_at_k(n, c, k) for c in counts) / total for k in ks}
    table_legacy = {f"pass@{k}": 100 * sum(pass_at_k(n, c, k) for c in counts_legacy) / total
                    for k in ks}

    # How many problems were solved at least once, and how often.
    buckets = Counter()
    for c in counts:
        if c == 0:
            buckets["0"] += 1
        elif c <= n // 8:
            buckets[f"1-{n // 8}"] += 1
        elif c <= n // 2:
            buckets[f"{n // 8 + 1}-{n // 2}"] += 1
        else:
            buckets[f"{n // 2 + 1}-{n}"] += 1

    hist = Counter(counts)

    # Samples where the two graders disagree: text with the right answer after
    # '####' but a different last number, or vice versa.
    total_samples = total * n
    fixed_only = sum(r["n_correct_only_fixed"] for r in rows)
    legacy_only = sum(r["n_correct_only_legacy"] for r in rows)
    uses_hash = sum(r["n_uses_hash"] for r in rows)

    return {
        "problems": total,
        "samples_per_problem": n,
        "pass_at_k": table,
        "pass_at_k_legacy_grader": table_legacy,
        "solve_count_buckets": dict(sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))),
        "solve_count_histogram": {str(k): hist[k] for k in sorted(hist)},
        "grader": {
            "samples": total_samples,
            "uses_hash_pct": 100 * uses_hash / total_samples,
            "correct_fixed_only": fixed_only,
            "correct_legacy_only": legacy_only,
        },
    }


def print_summary(s: dict):
    n = s["samples_per_problem"]
    print(f"\n{'=' * 60}")
    print(f"GSM8K pass@k   {s['problems']} problems x {n} samples")
    print(f"{'=' * 60}")
    print(f"{'k':>6} {'pass@k':>9} {'legacy grader':>15}")
    for key, v in s["pass_at_k"].items():
        print(f"{key[5:]:>6} {v:>8.2f}% {s['pass_at_k_legacy_grader'][key]:>14.2f}%")

    print("\nProblems by number of correct samples:")
    for b, cnt in s["solve_count_buckets"].items():
        print(f"  solved {b:>6} of {n}: {cnt:>5}  ({100 * cnt / s['problems']:.1f}%)")

    g = s["grader"]
    print(f"\nGrader: {g['uses_hash_pct']:.1f}% of samples use '####'. "
          f"Fixed grader alone marks {g['correct_fixed_only']} samples correct; "
          f"legacy alone marks {g['correct_legacy_only']}.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def hf_export_dir(ckpt: Path) -> Path:
    """Same naming and staleness rule as run_benchmarks_direct.py."""
    import torch
    import export_hf

    hf_dir = REPO_ROOT / "hf" / f"{ckpt.parent.name}_{ckpt.stem}"
    marker = hf_dir / "picotalk_export.json"
    if marker.exists() and marker.stat().st_mtime >= ckpt.stat().st_mtime:
        print(f"Using HF export {hf_dir}")
    else:
        print(f"Exporting {ckpt} -> {hf_dir} for vLLM...")
        export_hf.export(str(ckpt), str(hf_dir), torch.bfloat16)
    return hf_dir


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    ckpt = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{ckpt.parent.name}_{ckpt.stem}_{args.split}_n{args.n}"
    jsonl_path = out_dir / f"{tag}.jsonl"
    summary_path = out_dir / f"{tag}_summary.json"

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    items = [(i, ex["question"], gold_answer(ex["answer"])) for i, ex in enumerate(dataset)]
    items = [it for it in items if it[2]]

    done = load_done(jsonl_path)
    todo = [it for it in items if it[0] not in done]
    print(f"{len(items)} problems, {len(done)} already done, {len(todo)} to go -> {jsonl_path}")

    if todo:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        hf_dir = hf_export_dir(ckpt)

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        llm = LLM(
            model=str(hf_dir),
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        )
        sampling = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        eos = tokenizer.eos_token_id

        def encode(question: str) -> list:
            return [tokenizer.bos_token_id] + tokenizer.encode(
                f"[INST] {question.strip()} [/INST]", add_special_tokens=False
            )

        t0 = time.time()
        n_done = 0
        n_tokens = 0
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            prompts = [TokensPrompt(prompt_token_ids=encode(q)) for _, q, _ in batch]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)

            with open(jsonl_path, "a") as f:
                for (idx, question, gold), out in zip(batch, outputs):
                    samples = []
                    for o in out.outputs:
                        ids = list(o.token_ids)
                        if eos in ids:
                            ids = ids[:ids.index(eos)]
                        n_tokens += len(ids)
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        pred = extract_answer(text)
                        pred_legacy = extract_last_number(text)
                        samples.append({
                            "text": text,
                            "pred": pred,
                            "correct": is_correct(pred, gold),
                            "correct_legacy": is_correct(pred_legacy, gold),
                            "uses_hash": "####" in text,
                            "truncated": len(o.token_ids) >= args.max_new_tokens and eos not in o.token_ids,
                        })
                    row = {
                        "idx": idx,
                        "question": question,
                        "gold": gold,
                        "n": len(samples),
                        "n_correct": sum(s["correct"] for s in samples),
                        "n_correct_legacy": sum(s["correct_legacy"] for s in samples),
                        "n_correct_only_fixed": sum(s["correct"] and not s["correct_legacy"] for s in samples),
                        "n_correct_only_legacy": sum(s["correct_legacy"] and not s["correct"] for s in samples),
                        "n_uses_hash": sum(s["uses_hash"] for s in samples),
                        "n_truncated": sum(s["truncated"] for s in samples),
                        "samples": samples,
                    }
                    f.write(json.dumps(row) + "\n")

            n_done += len(batch)
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (len(todo) - n_done) / rate if rate > 0 else float("nan")
            print(f"[{time.strftime('%H:%M:%S')}] {len(done) + n_done}/{len(items)} problems  "
                  f"{n_tokens / elapsed:,.0f} tok/s  eta {eta / 60:.1f} min", flush=True)

    rows = [json.loads(line) for line in open(jsonl_path)]
    rows.sort(key=lambda r: r["idx"])
    n = rows[0]["n"]
    summary = {
        "checkpoint": str(ckpt),
        "split": args.split,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "truncated_samples": sum(r["n_truncated"] for r in rows),
        **summarize(rows, n),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nPer-problem results: {jsonl_path}")
    print(f"Summary:             {summary_path}")


def main():
    p = argparse.ArgumentParser(description="pass@k on GSM8K for a Picotalk SFT checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/sft_1b/final.pt")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n", type=int, default=32, help="samples per problem")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256, help="problems per vLLM call")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="first N problems only")
    p.add_argument("--output-dir", default="eval/results/gsm8k_passk")
    p.add_argument("--summarize", metavar="JSONL", default=None,
                   help="recompute the summary from an existing .jsonl and exit")
    args = p.parse_args()

    if args.summarize:
        rows = [json.loads(line) for line in open(args.summarize)]
        print_summary(summarize(rows, rows[0]["n"]))
        return

    run(args)


if __name__ == "__main__":
    main()
