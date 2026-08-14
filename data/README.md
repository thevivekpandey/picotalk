# Picotalk Data Pipeline

This directory contains scripts for downloading and preparing training data.

## Pipeline Overview

1. **Download** → Raw JSONL files (`./raw/`)
2. **Tokenize** → Binary numpy arrays (`./tokenized/`)
3. **Split** → Train/val datasets (`./tokenized/split/`)

## Usage

### 1. Download Data (currently running)

```bash
# Download all datasets (30B tokens, ~104GB)
python data/download_datasets.py --datasets all

# Or download individually
python data/download_datasets.py --datasets openwebmath
python data/download_datasets.py --datasets fineweb_edu
# etc.
```

**Output:** `data/raw/*.jsonl` files

### 2. Tokenize Data (after download completes)

```bash
# Tokenize all downloaded files
python data/tokenize_data.py

# Or specify paths
python data/tokenize_data.py --input-dir ./data/raw --output-dir ./data/tokenized
```

**Output:** `data/tokenized/*.npy` files (~60GB)

**Features:**
- Uses Mistral tokenizer (32k vocab)
- Efficient binary format (uint16)
- Skips already tokenized files
- Progress bars for tracking

### 3. Create Train/Val Split

```bash
# Create 99% train / 1% val split
python data/tokenize_data.py --create-split

# Or custom split
python data/tokenize_data.py --create-split --val-fraction 0.02  # 2% val
```

**Output:**
- `data/tokenized/split/train.npy` (~29.7B tokens)
- `data/tokenized/split/val.npy` (~300M tokens)

## Storage Management

**Disk space usage:**
- Raw JSONL: ~104GB
- Tokenized: ~60GB
- Total: ~164GB (within your 200GB budget)

**After tokenization, you can delete raw JSONL files to save space:**
```bash
# Only do this AFTER verifying tokenization worked!
rm data/raw/*.jsonl
# This frees up ~104GB
```

## Dataset Mix (30B tokens)

| Dataset | Tokens | Percentage | Size |
|---------|--------|------------|------|
| Nemotron-CC-VHQ | 15B | 50% | ~50GB |
| FineWeb-Edu | 6B | 20% | ~20GB |
| Stack v2 (code) | 6B | 20% | ~25GB |
| OpenWebMath | 2B | 7% | ~6GB |
| ArXiv | 1B | 3% | ~3GB |

## Tokenizer

**Using:** `mistralai/Mistral-7B-v0.1`
- Vocab size: 32,000 tokens
- Compression: ~3.6 chars/token
- Excellent for chat + code + reasoning

## Troubleshooting

**Out of disk space?**
- Delete cached HuggingFace downloads: `rm -rf ~/.cache/huggingface`
- Download datasets one at a time
- Tokenize and delete raw files incrementally

**Download interrupted?**
- Just re-run the command - it will skip already downloaded files

**Tokenization slow?**
- Normal! Expect 1-3 hours for 30B tokens
- Runs on CPU (no GPU needed)
