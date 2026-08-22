#!/usr/bin/env python3
"""
Prepare SFT data for Picotalk.

Two stages:

1. Download: each instruction dataset is downloaded IN FULL and saved as
   plain JSONL (one example per line, original schema) under data/sft_raw/.
   Files already present are not re-downloaded, so this is a one-time cost
   and the raw data stays inspectable on disk.

2. Tokenize: reads the local JSONL, converts each example to a unified
   messages format, tokenizes with the Mistral tokenizer using the [INST]
   chat template, and packs everything into memory-mapped .bin files
   compatible with the existing training setup:

    data/sft_raw/
        openhermes.jsonl       full raw dataset dumps
        slimorca.jsonl
        ...
    data/sft_tokenized/
        sft_train_tokens.bin   uint16 token ids (packed stream)
        sft_train_mask.bin     uint8 loss mask (1 = assistant token, train on it)
        sft_val_tokens.bin
        sft_val_mask.bin
        meta.json              stats and provenance

Usage:
    # Full mix (~205K tokenized examples, the default)
    python3 data/prepare_sft_data.py

    # Quick local smoke test (still downloads the full raw datasets once)
    python3 data/prepare_sft_data.py --limit 50

    # Single dataset
    python3 data/prepare_sft_data.py --datasets openhermes --limit 1000

    # Download the raw data only, skip tokenization
    python3 data/prepare_sft_data.py --download-only
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from data.chat_format import tokenize_conversation

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

# Conversations longer than this are skipped entirely (never truncated:
# a truncated response would train the model to stop mid-sentence).
MAX_TOKENS_PER_CONVERSATION = 2048

SEED = 1337


# ---------------------------------------------------------------------------
# Converters: dataset-specific example -> [{"role": ..., "content": ...}, ...]
# Return None to skip an example (bad roles, empty content, etc.)
# ---------------------------------------------------------------------------

_SHAREGPT_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def _from_sharegpt(conversations):
    """ShareGPT style: [{"from": "human", "value": ...}, ...]"""
    messages = []
    for turn in conversations:
        role = _SHAREGPT_ROLES.get(turn.get("from"))
        content = (turn.get("value") or "").strip()
        if role is None:
            return None  # unknown role -> skip whole conversation
        if not content:
            return None
        messages.append({"role": role, "content": content})
    return messages


def convert_openhermes(example):
    return _from_sharegpt(example["conversations"])


def convert_slimorca(example):
    return _from_sharegpt(example["conversations"])


def convert_messages(example):
    """OpenAI-style: {"messages": [{"role": ..., "content": ...}, ...]}
    Used by tulu, smoltalk, ultrachat, no_robots."""
    messages = []
    for msg in example["messages"]:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("system", "user", "assistant") or not content:
            return None
        messages.append({"role": role, "content": content})
    return messages


def convert_magpie(example):
    return _from_sharegpt(example["conversations"])


def convert_capybara(example):
    """Capybara: {"conversation": [{"input": ..., "output": ...}, ...]}"""
    messages = []
    for turn in example["conversation"]:
        user = (turn.get("input") or "").strip()
        assistant = (turn.get("output") or "").strip()
        if not user or not assistant:
            return None
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def convert_numinamath(example):
    return [
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["solution"]},
    ]


def convert_metamath(example):
    return [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]


def convert_magicoder(example):
    return [
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["solution"]},
    ]


def convert_orca_math(example):
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


# name -> (hf_repo, hf_config, split, converter, default_count)
DATASETS = {
    "openhermes": ("teknium/OpenHermes-2.5", None, "train", convert_openhermes, 15_000),
    "slimorca": ("Open-Orca/SlimOrca", None, "train", convert_slimorca, 20_000),
    "tulu": ("allenai/tulu-v2-sft-mixture", None, "train", convert_messages, 15_000),
    "magicoder": ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", convert_magicoder, 10_000),
    "metamath": ("meta-math/MetaMathQA", None, "train", convert_metamath, 10_000),
    "orca_math": ("microsoft/orca-math-word-problems-200k", None, "train", convert_orca_math, 5_000),
    # SmolLM2's SFT mix -- ablated specifically for 1-2B models; includes
    # Magpie-Ultra, instruction-following, rewriting/summarization subsets
    "smoltalk": ("HuggingFaceTB/smoltalk", "all", "train", convert_messages, 50_000),
    # Magpie self-synthesized from Llama-3-70B, quality-filtered
    "magpie": ("Magpie-Align/Magpie-Pro-300K-Filtered", None, "train", convert_magpie, 25_000),
    # Multi-turn dialogue (Zephyr's SFT data)
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", convert_messages, 15_000),
    # Human-written by professional annotators; small but highest quality
    "no_robots": ("HuggingFaceH4/no_robots", None, "train", convert_messages, 10_000),
    # Multi-turn, reasoning-dense conversations
    "capybara": ("LDJnr/Capybara", None, "train", convert_capybara, 15_000),
    # Competition-level math CoT (harder than metamath/orca_math)
    "numinamath": ("AI-MO/NuminaMath-CoT", None, "train", convert_numinamath, 15_000),
}


def is_valid(messages):
    """A usable conversation alternates and ends with an assistant reply."""
    if messages is None or len(messages) < 2:
        return False
    # After system-merge there must be at least one user->assistant exchange
    roles = [m["role"] for m in messages if m["role"] != "system"]
    if not roles or roles[-1] != "assistant" or "user" not in roles:
        return False
    return True


def download_dataset(name, hf_repo, hf_config, split, raw_dir):
    """Download the full dataset once and dump it as JSONL. Returns the path."""
    raw_file = raw_dir / f"{name}.jsonl"
    if raw_file.exists():
        size_gb = raw_file.stat().st_size / 1e9
        print(f"[{name}] raw file exists ({size_gb:.2f} GB), skipping download")
        return raw_file

    from datasets import load_dataset

    print(f"\n[{name}] downloading full dataset from {hf_repo} ...")
    if hf_config:
        ds = load_dataset(hf_repo, hf_config, split=split)
    else:
        ds = load_dataset(hf_repo, split=split)

    # Write to a temp name first so an interrupted dump is not mistaken for a
    # complete file on the next run.
    tmp_file = raw_file.with_suffix(".jsonl.tmp")
    ds.to_json(str(tmp_file), lines=True, force_ascii=False)
    tmp_file.rename(raw_file)

    size_gb = raw_file.stat().st_size / 1e9
    print(f"[{name}] saved {len(ds):,} examples -> {raw_file} ({size_gb:.2f} GB)")
    return raw_file


def collect_dataset(name, raw_file, converter, count, tokenizer):
    """Read local JSONL, returning up to `count` tokenized conversations."""
    print(f"\n[{name}] tokenizing from {raw_file} (target: {count:,} examples)")

    collected = []
    skipped_invalid = 0
    skipped_long = 0

    pbar = tqdm(total=count, desc=name)
    with open(raw_file) as f:
        for line in f:
            if len(collected) >= count:
                break

            example = json.loads(line)
            messages = converter(example)
            if not is_valid(messages):
                skipped_invalid += 1
                continue

            ids, mask = tokenize_conversation(messages, tokenizer)
            if len(ids) > MAX_TOKENS_PER_CONVERSATION:
                skipped_long += 1
                continue

            collected.append((ids, mask))
            pbar.update(1)
    pbar.close()

    print(f"[{name}] collected {len(collected):,} "
          f"(skipped: {skipped_invalid:,} invalid, {skipped_long:,} too long)")
    return collected


def write_split(examples, out_dir, split_name):
    """Pack (ids, mask) pairs into contiguous uint16/uint8 .bin files."""
    total = sum(len(ids) for ids, _ in examples)

    tokens = np.empty(total, dtype=np.uint16)
    mask = np.empty(total, dtype=np.uint8)

    pos = 0
    for ids, m in examples:
        n = len(ids)
        tokens[pos:pos + n] = ids
        mask[pos:pos + n] = m
        pos += n

    tokens_file = out_dir / f"sft_{split_name}_tokens.bin"
    mask_file = out_dir / f"sft_{split_name}_mask.bin"
    tokens.tofile(tokens_file)
    mask.tofile(mask_file)

    trainable = int(mask.sum())
    print(f"  {split_name}: {total:,} tokens ({trainable:,} trainable, "
          f"{trainable / total * 100:.1f}%) -> {tokens_file.name}")
    return {"tokens": total, "trainable_tokens": trainable, "examples": len(examples)}


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for Picotalk")
    parser.add_argument(
        "--datasets", type=str, default="all",
        help=f"Comma-separated subset of: {','.join(DATASETS)} (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Override per-dataset example count (for quick testing)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.02,
        help="Fraction of examples held out for validation (default: 0.02)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/sft_tokenized",
        help="Output directory for .bin files",
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/sft_raw",
        help="Directory for full raw JSONL dumps (kept on disk)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download the raw datasets and exit without tokenizing",
    )
    args = parser.parse_args()

    if args.datasets == "all":
        selected = list(DATASETS.keys())
    else:
        selected = args.datasets.split(",")
        for name in selected:
            if name not in DATASETS:
                parser.error(f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}")

    # Stage 1: download full raw datasets (skips files already present)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_files = {}
    for name in selected:
        hf_repo, hf_config, split, _, _ = DATASETS[name]
        raw_files[name] = download_dataset(name, hf_repo, hf_config, split, raw_dir)

    if args.download_only:
        print("\n--download-only set, stopping after download.")
        return

    # Stage 2: tokenize from the local raw files
    from transformers import AutoTokenizer
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    all_examples = []
    per_dataset_stats = {}
    for name in selected:
        _, _, _, converter, default_count = DATASETS[name]
        count = args.limit or default_count
        examples = collect_dataset(name, raw_files[name], converter, count, tokenizer)
        per_dataset_stats[name] = len(examples)
        all_examples.extend(examples)

    if not all_examples:
        print("No examples collected, nothing to write.")
        return

    # Shuffle so the packed stream interleaves datasets (a batch drawn from a
    # contiguous region would otherwise be single-dataset).
    random.Random(SEED).shuffle(all_examples)

    # Split train/val
    n_val = max(1, int(len(all_examples) * args.val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(train_examples):,} train / {len(val_examples):,} val examples...")
    train_stats = write_split(train_examples, out_dir, "train")
    val_stats = write_split(val_examples, out_dir, "val")

    meta = {
        "tokenizer": TOKENIZER_NAME,
        "chat_format": "mistral_inst",
        "max_tokens_per_conversation": MAX_TOKENS_PER_CONVERSATION,
        "seed": SEED,
        "raw_files": {name: str(path) for name, path in raw_files.items()},
        "datasets": per_dataset_stats,
        "train": train_stats,
        "val": val_stats,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Metadata written to {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
