#!/usr/bin/env python3
"""
Inspect tokenized data by sampling and decoding random segments.

For each tokenized .bin file, generates 10 sample .txt files with 10K decoded tokens each.
This helps verify that tokenization was done correctly.
"""

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


def sample_and_decode(
    bin_file: Path,
    tokenizer,
    num_samples: int = 10,
    tokens_per_sample: int = 10000,
    output_dir: Path = None,
) -> List[Path]:
    """
    Sample random segments from a tokenized .bin file and decode them.

    Args:
        bin_file: Path to .bin file
        tokenizer: HuggingFace tokenizer
        num_samples: Number of samples to generate
        tokens_per_sample: Number of tokens per sample
        output_dir: Output directory for .txt files

    Returns:
        List of output file paths
    """
    if output_dir is None:
        output_dir = bin_file.parent.parent / "inspection"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the memory-mapped array
    tokens = np.memmap(bin_file, dtype=np.uint16, mode='r')
    total_tokens = len(tokens)

    print(f"\n{'='*60}")
    print(f"Processing: {bin_file.name}")
    print(f"{'='*60}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Generating {num_samples} samples of {tokens_per_sample:,} tokens each")

    if total_tokens < tokens_per_sample:
        print(f"⚠️  File too small ({total_tokens} tokens), skipping...")
        return []

    output_files = []

    for i in range(num_samples):
        # Choose random start point
        max_start = total_tokens - tokens_per_sample
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + tokens_per_sample

        # Extract tokens
        sample_tokens = tokens[start_idx:end_idx]

        # Convert to list for decoding
        token_ids = sample_tokens.astype(np.int64).tolist()

        # Decode
        try:
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception as e:
            print(f"⚠️  Error decoding sample {i+1}: {e}")
            continue

        # Create output filename
        dataset_name = bin_file.stem.replace('_train', '').replace('_val', '')
        split = 'train' if '_train' in bin_file.stem else 'val'
        output_file = output_dir / f"{dataset_name}_{split}_sample_{i+1:02d}.txt"

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Dataset: {dataset_name}\n")
            f.write(f"# Split: {split}\n")
            f.write(f"# Source file: {bin_file.name}\n")
            f.write(f"# Start token index: {start_idx:,}\n")
            f.write(f"# End token index: {end_idx:,}\n")
            f.write(f"# Number of tokens: {tokens_per_sample:,}\n")
            f.write(f"# Number of characters: {len(text):,}\n")
            f.write(f"{'='*60}\n\n")
            f.write(text)

        output_files.append(output_file)

        # Show preview
        if i == 0:
            preview = text[:200].replace('\n', '\\n')
            print(f"\nSample 1 preview (first 200 chars):")
            print(f"  {preview}...")

    print(f"\n✓ Generated {len(output_files)} samples")
    print(f"  Output directory: {output_dir}")

    return output_files


def main():
    parser = argparse.ArgumentParser(description="Inspect tokenized data")

    parser.add_argument(
        '--input-dir',
        type=str,
        default='./data/tokenized',
        help='Directory with tokenized .bin files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/inspection',
        help='Output directory for decoded samples'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples per file (default: 10)'
    )
    parser.add_argument(
        '--tokens-per-sample',
        type=int,
        default=10000,
        help='Number of tokens per sample (default: 10,000)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Specific datasets to inspect (default: all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("Tokenized Data Inspection")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per file: {args.num_samples}")
    print(f"Tokens per sample: {args.tokens_per_sample:,}")
    print(f"Random seed: {args.seed}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")

    # Find all .bin files
    if args.datasets:
        bin_files = []
        for dataset in args.datasets:
            bin_files.extend(list(input_dir.glob(f"{dataset}_*.bin")))
    else:
        bin_files = sorted(input_dir.glob("*_train.bin"))

    if not bin_files:
        print(f"\n✗ No .bin files found in {input_dir}")
        return

    print(f"\nFound {len(bin_files)} files to process:")
    for f in bin_files:
        size_mb = f.stat().st_size / 1024**2
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Process each file
    all_outputs = []
    for bin_file in bin_files:
        outputs = sample_and_decode(
            bin_file,
            tokenizer,
            num_samples=args.num_samples,
            tokens_per_sample=args.tokens_per_sample,
            output_dir=output_dir,
        )
        all_outputs.extend(outputs)

    # Summary
    print(f"\n{'='*60}")
    print("Inspection Summary")
    print(f"{'='*60}")
    print(f"Total samples generated: {len(all_outputs)}")
    print(f"Output directory: {output_dir}")

    # Group by dataset
    datasets = {}
    for f in all_outputs:
        dataset = f.stem.rsplit('_sample_', 1)[0]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(f)

    print(f"\nSamples per dataset:")
    for dataset, files in sorted(datasets.items()):
        print(f"  {dataset}: {len(files)} samples")

    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Review the generated .txt files in the inspection directory")
    print("2. Check for:")
    print("   - Proper text formatting (no weird characters)")
    print("   - Sensible content (matches expected domain)")
    print("   - EOS tokens appear as separators between documents")
    print("   - No obvious tokenization errors")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
