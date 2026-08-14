#!/usr/bin/env python3
"""
Tokenize downloaded datasets using Mistral tokenizer.

Design (built for 30B tokens on a laptop):
- Tokens are streamed to disk as raw uint16 .bin files (never held in RAM).
- Train/val split happens at DOCUMENT level during tokenization
  (every Nth document goes to val). No post-hoc shuffle of tokens.
- An EOS token is appended after every document as a separator.
- Each dataset is processed independently, so raw .jsonl files can be
  deleted as soon as their .bin files are verified.

Output layout:
  data/tokenized/{name}_train.bin   uint16 token stream
  data/tokenized/{name}_val.bin     uint16 token stream
  data/tokenized/{name}_stats.json  token counts etc.

Reading later:  np.memmap(path, dtype=np.uint16, mode='r')
"""

import ast
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


class DataTokenizer:
    """Tokenize text data and stream to raw uint16 binary files."""

    def __init__(self, tokenizer_name: str = "mistralai/Mistral-7B-v0.1"):
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_id = self.tokenizer.eos_token_id
        assert self.eos_id is not None, "Tokenizer must define an EOS token"
        assert self.vocab_size <= 65535, "uint16 storage requires vocab <= 65535"
        print(f"✓ Tokenizer loaded (vocab size: {self.vocab_size:,}, eos id: {self.eos_id})")

    def tokenize_file(
        self,
        input_file: Path,
        output_dir: Path,
        val_every_n_docs: int = 100,
        batch_docs: int = 1000,
    ) -> Optional[Dict]:
        """
        Tokenize a JSONL file, streaming tokens into train/val .bin files.

        Every `val_every_n_docs`-th document goes to the val stream (~1%).
        """
        name = input_file.stem
        train_file = output_dir / f"{name}_train.bin"
        val_file = output_dir / f"{name}_val.bin"
        stats_file = output_dir / f"{name}_stats.json"

        print(f"\n{'='*60}")
        print(f"Tokenizing: {input_file.name}")
        print(f"{'='*60}")

        if not input_file.exists():
            print(f"✗ File not found: {input_file}")
            return None

        if stats_file.exists() and train_file.exists():
            print(f"✓ Already tokenized (found {stats_file.name}), skipping.")
            print(f"  Delete {stats_file.name} to re-tokenize.")
            with open(stats_file) as f:
                return json.load(f)

        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'dataset': name,
            'train_tokens': 0,
            'val_tokens': 0,
            'train_docs': 0,
            'val_docs': 0,
            'total_chars': 0,
        }

        doc_index = 0
        batch_texts: List[str] = []
        batch_is_val: List[bool] = []

        def flush(train_f, val_f):
            """Tokenize the current batch and append to the bin files."""
            if not batch_texts:
                return
            encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            train_ids: List[int] = []
            val_ids: List[int] = []
            for ids, is_val in zip(encoded['input_ids'], batch_is_val):
                target = val_ids if is_val else train_ids
                target.extend(ids)
                target.append(self.eos_id)
                if is_val:
                    stats['val_docs'] += 1
                    stats['val_tokens'] += len(ids) + 1
                else:
                    stats['train_docs'] += 1
                    stats['train_tokens'] += len(ids) + 1
            if train_ids:
                np.asarray(train_ids, dtype=np.uint16).tofile(train_f)
            if val_ids:
                np.asarray(val_ids, dtype=np.uint16).tofile(val_f)
            batch_texts.clear()
            batch_is_val.clear()

        try:
            with open(input_file, 'r') as fin, \
                 open(train_file, 'wb') as train_f, \
                 open(val_file, 'wb') as val_f:

                pbar = tqdm(fin, desc=name, unit=" docs")
                for line in pbar:
                    text = self._extract_text(line, name)
                    if not text:
                        continue
                    stats['total_chars'] += len(text)
                    batch_texts.append(text)
                    batch_is_val.append(doc_index % val_every_n_docs == 0)
                    doc_index += 1

                    if len(batch_texts) >= batch_docs:
                        flush(train_f, val_f)
                        pbar.set_postfix(
                            tokens=f"{(stats['train_tokens'] + stats['val_tokens'])/1e9:.2f}B"
                        )

                flush(train_f, val_f)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted. Partial .bin files kept; stats NOT saved,")
            print("    so a re-run will start this dataset from scratch.")
            return None

        total = stats['train_tokens'] + stats['val_tokens']
        stats['chars_per_token'] = stats['total_chars'] / total if total else 0
        stats['train_size_mb'] = train_file.stat().st_size / 1024**2
        stats['val_size_mb'] = val_file.stat().st_size / 1024**2

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ {name} complete:")
        print(f"  Train: {stats['train_tokens']:,} tokens "
              f"({stats['train_docs']:,} docs, {stats['train_size_mb']:.0f} MB)")
        print(f"  Val:   {stats['val_tokens']:,} tokens "
              f"({stats['val_docs']:,} docs, {stats['val_size_mb']:.0f} MB)")
        print(f"  Chars/token: {stats['chars_per_token']:.2f}")
        print(f"\n  Raw file can now be deleted:  rm {input_file}")

        return stats

    # Dataset-specific field mappings
    FIELD_MAP = {
        'arxiv': 'text',
        'cosmopedia': 'text',
        'dclm': 'text',
        'fineweb_edu': 'text',
        'openwebmath': 'text',
        'stack_v2': 'content',
    }

    def _extract_text(self, line: str, dataset_name: str) -> str:
        """Extract the text field from a raw line (JSON format)."""
        line = line.strip()
        if not line:
            return ""

        field = self.FIELD_MAP.get(dataset_name, 'text')
        data = json.loads(line)
        return data.get(field, '')


def verify_bin(path: Path, vocab_size: int, n_check: int = 1_000_000) -> bool:
    """Sanity-check a .bin file: readable, sane length, ids < vocab size."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.stat().st_size % 2 != 0:
        return False
    arr = np.memmap(path, dtype=np.uint16, mode='r')
    head = np.asarray(arr[:n_check])
    return bool((head < vocab_size + 1000).all())  # allow special tokens above base vocab


def main():
    parser = argparse.ArgumentParser(description="Tokenize Picotalk training data")
    parser.add_argument("--input-dir", type=str, default="./data/raw")
    parser.add_argument("--output-dir", type=str, default="./data/tokenized")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset names to tokenize (e.g. arxiv dclm). Default: all *.jsonl found."
    )
    parser.add_argument(
        "--val-every", type=int, default=100,
        help="Every Nth document goes to val (default 100 -> ~1%%)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("Picotalk Data Tokenization")
    print(f"{'='*60}")
    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"Tokenizer: {args.tokenizer}")

    if args.datasets:
        jsonl_files = [input_dir / f"{n}.jsonl" for n in args.datasets]
    else:
        jsonl_files = sorted(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"\n✗ No .jsonl files found in {input_dir}")
        return

    print(f"\nDatasets to process: {[f.stem for f in jsonl_files]}\n")

    tokenizer = DataTokenizer(args.tokenizer)

    all_stats = {}
    for jsonl_file in jsonl_files:
        stats = tokenizer.tokenize_file(
            jsonl_file, output_dir, val_every_n_docs=args.val_every
        )
        if stats:
            all_stats[jsonl_file.stem] = stats
            # Verify output
            ok = verify_bin(output_dir / f"{jsonl_file.stem}_train.bin",
                            tokenizer.vocab_size)
            print(f"  Verification: {'✓ passed' if ok else '✗ FAILED - do not delete raw!'}")

    # Summary
    print(f"\n{'='*60}")
    print("Tokenization Summary")
    print(f"{'='*60}")
    total_train = total_val = 0
    for name, s in all_stats.items():
        total_train += s['train_tokens']
        total_val += s['val_tokens']
        print(f"  {name:<15} train {s['train_tokens']/1e9:>6.2f}B   "
              f"val {s['val_tokens']/1e6:>7.1f}M")
    print(f"\n  Total train: {total_train:,} ({total_train/1e9:.2f}B)")
    print(f"  Total val:   {total_val:,} ({total_val/1e6:.1f}M)")
    print(f"  Disk: ~{2*(total_train+total_val)/1024**3:.1f} GB of .bin files")


if __name__ == "__main__":
    main()
