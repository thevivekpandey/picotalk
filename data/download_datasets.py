#!/usr/bin/env python3
"""
Download and prepare datasets for Picotalk training.
Optimized for limited disk space - streams and processes incrementally.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import yaml
import argparse
from typing import Optional
import psutil
import json

def get_disk_space_gb(path: str = ".") -> float:
    """Get available disk space in GB."""
    stat = psutil.disk_usage(path)
    return stat.free / (1024**3)

def download_generic(
    dataset_key: str,
    config: dict,
    output_dir: Path,
    max_tokens: Optional[int] = None,
    text_field: str = "text",
):
    """
    Generic downloader for streaming HF datasets (parquet-based, no scripts).
    Used for DCLM, Cosmopedia, and other standard datasets.
    """
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_key}: {config['name']}")
    if config.get('subset'):
        print(f"Subset: {config['subset']}")
    print(f"Target: {max_tokens/1e9:.1f}B tokens")
    print(f"{'='*60}\n")

    output_file = output_dir / f"{dataset_key}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if output_file.exists():
        print(f"✓ File already exists: {output_file}")
        print(f"  Skipping download. Delete the file to re-download.")
        return output_file

    try:
        subset = config.get('subset')

        if subset:
            dataset = load_dataset(
                config['name'],
                subset,
                split="train",
                streaming=True
            )
        else:
            dataset = load_dataset(
                config['name'],
                split="train",
                streaming=True
            )

        print(f"Streaming and saving to: {output_file}")
        print("This will take a while. Press Ctrl+C to stop early.\n")

        token_count = 0
        doc_count = 0

        with open(output_file, 'w') as f:
            for example in tqdm(dataset, desc=f"Processing {dataset_key}"):
                # Estimate tokens (rough: 1 token ≈ 4 chars)
                text = example.get(text_field, '')

                if not text:
                    continue

                estimated_tokens = len(text) // 4

                if max_tokens and token_count + estimated_tokens > max_tokens:
                    print(f"\nReached target token count: {token_count:,}")
                    break

                f.write(json.dumps(example) + '\n')
                token_count += estimated_tokens
                doc_count += 1

                if doc_count % 10000 == 0:
                    print(f"  Downloaded {doc_count:,} docs, ~{token_count/1e9:.2f}B tokens")
                    print(f"  Disk space remaining: {get_disk_space_gb():.1f} GB")

        print(f"\n✓ {dataset_key} complete: {doc_count:,} documents, ~{token_count/1e9:.2f}B tokens")
        return output_file

    except Exception as e:
        print(f"Error downloading {dataset_key}: {e}")
        print("Note: You may need to authenticate with HuggingFace CLI first:")
        print("  huggingface-cli login")
        return None

def download_fineweb_edu(config: dict, output_dir: Path, max_tokens: Optional[int] = None):
    """Download FineWeb-Edu subset."""
    print(f"\n{'='*60}")
    print(f"Downloading FineWeb-Edu")
    print(f"Target: {max_tokens/1e9:.1f}B tokens")
    print(f"{'='*60}\n")

    output_file = output_dir / "fineweb_edu.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if output_file.exists():
        print(f"✓ File already exists: {output_file}")
        print(f"  Skipping download. Delete the file to re-download.")
        return output_file

    try:
        # FineWeb-Edu has a 10BT sample which is perfect for us
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True
        )

        token_count = 0
        doc_count = 0

        with open(output_file, 'w') as f:
            for example in tqdm(dataset, desc="Processing FineWeb-Edu"):
                text = example.get('text', '')
                estimated_tokens = len(text) // 4

                if max_tokens and token_count + estimated_tokens > max_tokens:
                    break

                f.write(json.dumps(example) + '\n')
                token_count += estimated_tokens
                doc_count += 1

                if doc_count % 10000 == 0:
                    print(f"  Downloaded {doc_count:,} docs, ~{token_count/1e9:.2f}B tokens")

        print(f"\n✓ FineWeb-Edu complete: {doc_count:,} documents, ~{token_count/1e9:.2f}B tokens")
        return output_file

    except Exception as e:
        print(f"Error downloading FineWeb-Edu: {e}")
        return None

def download_stack_v2(config: dict, output_dir: Path, max_tokens: Optional[int] = None):
    """Download The Stack v2 (filtered languages)."""
    print(f"\n{'='*60}")
    print(f"Downloading Code Dataset: {config['name']}")
    if 'languages' in config:
        print(f"Languages: {config['languages']}")
    if 'subset' in config:
        print(f"Subset: {config['subset']}")
    print(f"Target: {max_tokens/1e9:.1f}B tokens")
    print(f"{'='*60}\n")

    output_file = output_dir / "stack_v2.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if output_file.exists():
        print(f"✓ File already exists: {output_file}")
        print(f"  Skipping download. Delete the file to re-download.")
        return output_file

    try:
        token_count = 0
        doc_count = 0

        # Check if this is StarCoderData
        if 'starcoderdata' in config['name']:
            # StarCoderData format - has actual content
            print(f"Using StarCoderData...")
            subset = config.get('subset', 'python')

            dataset = load_dataset(
                config['name'],
                data_files=f"{subset}/*",
                split="train",
                streaming=True
            )

            with open(output_file, 'w') as f:
                for example in tqdm(dataset, desc="Processing Python code"):
                    text = example.get('content', '')

                    if not text or len(text) < 50:  # Skip very short files
                        continue

                    estimated_tokens = len(text) // 4

                    if max_tokens and token_count + estimated_tokens > max_tokens:
                        print(f"\nReached target: {token_count:,} tokens")
                        break

                    f.write(json.dumps(example) + '\n')
                    token_count += estimated_tokens
                    doc_count += 1

                    if doc_count % 5000 == 0:
                        print(f"  {doc_count:,} docs, ~{token_count/1e9:.2f}B tokens")

        else:
            # Stack v2 format
            languages = config.get('languages', ['Python'])
            tokens_per_lang = max_tokens // len(languages) if max_tokens else None

            for lang in languages:
                print(f"\nDownloading {lang}...")

                try:
                    # Stack v2 structure uses subset parameter
                    dataset = load_dataset(
                        config['name'],
                        data_dir=f"data/{lang}",
                        split="train",
                        streaming=True
                    )

                    lang_tokens = 0
                    with open(output_file, 'a') as f:
                        for example in tqdm(dataset, desc=f"Processing {lang}"):
                            text = example.get('content', example.get('text', ''))
                            estimated_tokens = len(text) // 4

                            if tokens_per_lang and lang_tokens + estimated_tokens > tokens_per_lang:
                                break

                            f.write(json.dumps(example) + '\n')
                            token_count += estimated_tokens
                            lang_tokens += estimated_tokens
                            doc_count += 1

                            if doc_count % 5000 == 0:
                                print(f"    {doc_count:,} docs, ~{token_count/1e9:.2f}B tokens")

                    print(f"  {lang}: ~{lang_tokens/1e9:.2f}B tokens")

                except Exception as e:
                    print(f"  Error: {e}")
                    continue

        print(f"\n✓ Stack v2 complete: {doc_count:,} documents, ~{token_count/1e9:.2f}B tokens")
        return output_file

    except Exception as e:
        print(f"Error downloading Stack v2: {e}")
        return None

def download_openwebmath(config: dict, output_dir: Path, max_tokens: Optional[int] = None):
    """Download OpenWebMath."""
    print(f"\n{'='*60}")
    print(f"Downloading OpenWebMath")
    print(f"Target: {max_tokens/1e9:.1f}B tokens")
    print(f"{'='*60}\n")

    output_file = output_dir / "openwebmath.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if output_file.exists():
        print(f"✓ File already exists: {output_file}")
        print(f"  Skipping download. Delete the file to re-download.")
        return output_file

    try:
        dataset = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True
        )

        token_count = 0
        doc_count = 0

        with open(output_file, 'w') as f:
            for example in tqdm(dataset, desc="Processing OpenWebMath"):
                text = example.get('text', '')
                estimated_tokens = len(text) // 4

                if max_tokens and token_count + estimated_tokens > max_tokens:
                    break

                f.write(json.dumps(example) + '\n')
                token_count += estimated_tokens
                doc_count += 1

        print(f"\n✓ OpenWebMath complete: {doc_count:,} documents, ~{token_count/1e9:.2f}B tokens")
        return output_file

    except Exception as e:
        print(f"Error downloading OpenWebMath: {e}")
        return None

def download_arxiv(config: dict, output_dir: Path, max_tokens: Optional[int] = None):
    """Download ArXiv papers from proof-pile-2 (RedPajama ArXiv subset)."""
    print(f"\n{'='*60}")
    print(f"Downloading ArXiv (from {config['name']})")
    print(f"Target: {max_tokens/1e9:.1f}B tokens")
    print(f"{'='*60}\n")

    output_file = output_dir / "arxiv.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if output_file.exists():
        print(f"✓ File already exists: {output_file}")
        print(f"  Skipping download. Delete the file to re-download.")
        return output_file

    try:
        # proof-pile-2 has a loading script, so we bypass it by using the
        # generic 'json' builder pointed directly at the arxiv jsonl.zst files
        dataset = load_dataset(
            "json",
            data_files=f"hf://datasets/{config['name']}/arxiv/train/*.jsonl.zst",
            split="train",
            streaming=True
        )

        token_count = 0
        doc_count = 0

        with open(output_file, 'w') as f:
            for example in tqdm(dataset, desc="Processing ArXiv"):
                text = example.get('text', '')
                estimated_tokens = len(text) // 4

                if max_tokens and token_count + estimated_tokens > max_tokens:
                    print(f"\nReached target token count: {token_count:,}")
                    break

                # Convert datetime objects to strings for JSON serialization
                if 'meta' in example and isinstance(example['meta'], dict):
                    if 'timestamp' in example['meta']:
                        example['meta']['timestamp'] = str(example['meta']['timestamp'])

                f.write(json.dumps(example) + '\n')
                token_count += estimated_tokens
                doc_count += 1

                if doc_count % 5000 == 0:
                    print(f"  {doc_count:,} docs, ~{token_count/1e9:.2f}B tokens")

        print(f"\n✓ ArXiv complete: {doc_count:,} documents, ~{token_count/1e9:.2f}B tokens")
        return output_file

    except Exception as e:
        print(f"Error downloading ArXiv: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download Picotalk training data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["dclm", "cosmopedia", "fineweb_edu", "stack_v2", "openwebmath", "arxiv", "all"],
        default=["all"],
        help="Which datasets to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory for downloaded data"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check disk space
    available_gb = get_disk_space_gb()
    print(f"\n{'='*60}")
    print(f"Picotalk Data Download")
    print(f"{'='*60}")
    print(f"Available disk space: {available_gb:.1f} GB")
    print(f"Estimated required: ~100-150 GB")

    if available_gb < 100:
        print("\n⚠️  WARNING: Low disk space! You may need to free up space.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    print(f"\nOutput directory: {output_dir}")
    print(f"Datasets to download: {args.datasets}")
    print(f"\nStarting downloads...\n")

    datasets_to_download = args.datasets if "all" not in args.datasets else [
        "dclm", "cosmopedia", "fineweb_edu", "stack_v2", "openwebmath", "arxiv"
    ]

    results = {}

    for dataset_name in datasets_to_download:
        if dataset_name == "all":
            continue

        dataset_config = config['datasets'].get(dataset_name)
        if not dataset_config:
            print(f"Warning: No config found for {dataset_name}, skipping...")
            continue

        max_tokens = dataset_config.get('target_tokens')

        # Call appropriate download function
        if dataset_name in ("dclm", "cosmopedia"):
            result = download_generic(dataset_name, dataset_config, output_dir, max_tokens)
        elif dataset_name == "fineweb_edu":
            result = download_fineweb_edu(dataset_config, output_dir, max_tokens)
        elif dataset_name == "stack_v2":
            result = download_stack_v2(dataset_config, output_dir, max_tokens)
        elif dataset_name == "openwebmath":
            result = download_openwebmath(dataset_config, output_dir, max_tokens)
        elif dataset_name == "arxiv":
            result = download_arxiv(dataset_config, output_dir, max_tokens)

        results[dataset_name] = result

        # Check disk space after each download
        remaining = get_disk_space_gb()
        print(f"\nDisk space remaining: {remaining:.1f} GB\n")

        if remaining < 20:
            print("⚠️  WARNING: Less than 20GB remaining!")
            print("Consider stopping and processing what you have so far.")

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for dataset_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {dataset_name}: {result if result else 'Failed'}")

    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Train tokenizer: python data/train_tokenizer.py")
    print("2. Tokenize datasets: python data/tokenize_data.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
