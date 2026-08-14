# Picotalk Training Guide

Complete training pipeline for Picotalk - from data download to final model.

## Architecture

**Final 1B Model (1.07B parameters)**:
- 30 layers (depth-focused for reasoning/code)
- 1792 embedding dimension
- 16 attention heads, 4 KV heads (GQA)
- 4778 FFN hidden dimension (2.67x ratio)
- 2048 context length
- Llama-style: RoPE, RMSNorm, SwiGLU, GQA
- Mistral tokenizer (32k vocab)

## Pipeline Overview

```
1. Download datasets (30B tokens)
   ↓
2. Tokenize to .bin files
   ↓
3. Run scaling experiments (50M-350M models)
   ↓
4. Train final 1B model
   ↓
5. Evaluate and fine-tune
```

## 1. Data Download

Download 30B token mix optimized for chat + reasoning + code:

```bash
# Download all datasets (~150GB disk space)
python3 data/download_datasets.py --datasets all

# Or download individually:
python3 data/download_datasets.py --datasets dclm cosmopedia fineweb_edu stack_v2 openwebmath arxiv
```

**Data Mix (30B tokens)**:
- DCLM: 10B tokens (33%) - high-quality web text
- Cosmopedia: 5B tokens (17%) - synthetic educational content
- FineWeb-Edu: 6B tokens (20%) - educational web pages
- StarCoder Python: 6B tokens (20%) - Python code
- OpenWebMath: 2B tokens (7%) - mathematical content
- ArXiv: 1B tokens (3%) - scientific papers

**Space Management**:
- Raw .jsonl files can be deleted after tokenization
- Keep only tokenized .bin files (~60GB total)

## 2. Tokenization

Convert raw text to uint16 binary token streams:

```bash
# Tokenize all datasets
python3 data/tokenize_data.py

# Or tokenize specific datasets:
python3 data/tokenize_data.py --datasets arxiv dclm

# Custom tokenizer (default: mistralai/Mistral-7B-v0.1):
python3 data/tokenize_data.py --tokenizer mistralai/Mistral-7B-v0.1
```

**Output Format**:
- `{dataset}_train.bin`: uint16 binary (memory-mapped)
- `{dataset}_val.bin`: validation split (~1% of data)
- `{dataset}_stats.json`: token counts, metadata
- EOS token inserted after each document
- No BOS token (GPT-style)

**Verification**:
```bash
# Check tokenized data
python3 -c "
import numpy as np
data = np.memmap('data/tokenized/arxiv_train.bin', dtype=np.uint16, mode='r')
print(f'Tokens: {len(data):,}')
print(f'Sample: {data[:20]}')
"
```

## 3. Scaling Experiments

Empirically derive scaling laws before committing to final training:

```bash
# List available experiments
python3 train/scaling_experiments.py --list

# Run specific experiments
python3 train/scaling_experiments.py --experiments 100m_3b 200m_5b

# Dry run to see commands
python3 train/scaling_experiments.py --experiments 50m_1b --dry-run
```

**Experiment Matrix**:
| Experiment | Params | Tokens | Steps | Batch | Est. Time (H100) |
|------------|--------|--------|-------|-------|------------------|
| 50m_1b     | 50M    | 1B     | 1K    | 128   | ~15 min          |
| 100m_3b    | 100M   | 3B     | 3K    | 120   | ~1.5 hours       |
| 200m_5b    | 200M   | 5B     | 5K    | 128   | ~3 hours         |
| 350m_10b   | 350M   | 10B    | 10K   | 120   | ~8 hours         |

**Goal**: Find optimal (model_size, data_size) to predict 1B model performance.

## 4. Main Training

### Quick Start (Single GPU)

```bash
# Train 1B model with default settings
python3 train/train.py \
  --model 1b \
  --data-dir ./data/tokenized \
  --batch-size 8 \
  --grad-accum-steps 16 \
  --max-steps 30000 \
  --wandb \
  --run-name picotalk-1b-30B
```

**Expected**:
- Effective batch size: 8 × 16 × 2048 = 262K tokens/step
- Total tokens: 30K steps × 262K = ~7.8B tokens
- Training time: ~40-50 hours on 1x H100

### Multi-GPU Training (4x or 8x H100)

```bash
# 4x H100 (torchrun)
torchrun --nproc_per_node=4 train/train.py \
  --model 1b \
  --data-dir ./data/tokenized \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --max-steps 30000 \
  --ddp \
  --wandb \
  --run-name picotalk-1b-30B-4gpu

# 8x H100
torchrun --nproc_per_node=8 train/train.py \
  --model 1b \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --max-steps 30000 \
  --ddp \
  --wandb \
  --run-name picotalk-1b-30B-8gpu
```

**Effective Batch Sizes**:
- 1x GPU: 8 × 16 = 128 sequences
- 4x GPU: 8 × 4 × 4 = 128 sequences
- 8x GPU: 8 × 2 × 8 = 128 sequences

### Training Options

```bash
python3 train/train.py --help
```

**Key Arguments**:
- `--model`: Model size (test, 50m, 100m, 200m, 350m, 1b)
- `--batch-size`: Per-device batch size
- `--grad-accum-steps`: Gradient accumulation steps
- `--max-steps`: Total training steps
- `--learning-rate`: Peak LR (default: 3e-4)
- `--min-lr`: Min LR for cosine decay (default: 3e-5)
- `--weight-decay`: Weight decay (default: 0.1)
- `--grad-clip`: Gradient clipping (default: 1.0)
- `--warmup-steps`: Warmup steps (default: max_steps/20)
- `--eval-interval`: Eval every N steps (default: 500)
- `--save-interval`: Save every N steps (default: 2000)
- `--wandb`: Enable wandb logging

### Resume Training

```bash
python3 train/train.py \
  --model 1b \
  --resume ./checkpoints/step_10000.pt
```

## 5. Evaluation

### Perplexity

```bash
# Evaluate on validation set
python3 eval/evaluate.py \
  ./checkpoints/best.pt \
  --perplexity ./data/validation.txt
```

### Text Generation

```bash
# Generate sample texts
python3 eval/evaluate.py \
  ./checkpoints/best.pt \
  --generate \
  --temperature 0.8 \
  --max-tokens 200

# Use custom prompts
python3 eval/evaluate.py \
  ./checkpoints/best.pt \
  --generate \
  --prompts ./prompts.txt \
  --temperature 0.7
```

## 6. Checkpoints

Checkpoint structure:
```python
{
  'step': int,                     # Training step
  'epoch': int,                    # Epoch number
  'tokens_seen': int,              # Total tokens processed
  'model_state_dict': dict,        # Model weights
  'optimizer_state_dict': dict,    # Optimizer state
  'scaler_state_dict': dict,       # AMP scaler state
  'config': dict,                  # Training config
  'model_config': dict,            # Model config
  'best_val_loss': float,          # Best validation loss
}
```

**Checkpoint Files**:
- `step_{N}.pt`: Regular checkpoint every N steps
- `best.pt`: Best model by validation loss
- `final.pt`: Final model after training

## 7. Monitoring with Wandb

Enable wandb logging:
```bash
# Login to wandb
wandb login

# Train with logging
python3 train/train.py \
  --model 1b \
  --wandb \
  --wandb-project picotalk \
  --run-name my-experiment
```

**Logged Metrics**:
- `train/loss`: Training loss
- `train/lr`: Learning rate
- `train/tokens_per_sec`: Throughput
- `train/tokens_seen`: Cumulative tokens
- `val/loss`: Validation loss
- `val/perplexity`: Validation perplexity

## 8. Compute Budget

**Target**: $2000 budget (~500 H100 hours)

**Scaling Experiments** (~50 H100 hours):
- 50M models: ~5 hours
- 100M models: ~10 hours
- 200M models: ~15 hours
- 350M models: ~20 hours

**Final 1B Training** (~400 H100 hours):
- Option A: 1x H100 for 400 hours (~17 days)
- Option B: 4x H100 for 100 hours (~4 days)
- Option C: 8x H100 for 50 hours (~2 days)

**Reserve** (~50 hours):
- Fine-tuning, RL, experiments

## 9. Expected Performance

Based on scaling laws and similar models:

**1B model @ 30B tokens**:
- Validation perplexity: ~15-20 (estimated)
- Coherent text generation
- Basic reasoning capabilities
- Python code generation (simple functions)
- Instruction following (after SFT)

**Comparison Models**:
- SmolLM (135M-1.7B): Similar architecture
- TinyLlama (1.1B): 3T tokens, stronger baseline
- Pythia (1B): Reference for scaling

## 10. Next Steps

After pretraining:

1. **Supervised Fine-Tuning (SFT)**:
   - Instruction-following datasets
   - Chat formatting
   - ~5B tokens, ~2-3 epochs

2. **Reinforcement Learning**:
   - DPO/RLHF on preferences
   - ~1B tokens
   - Human evaluation

3. **Deployment**:
   - Quantization (int8/int4)
   - GGUF export for llama.cpp
   - Inference optimization

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 4 --grad-accum-steps 32

# Disable mixed precision (slower)
--no-amp
```

### Slow Training
```bash
# Check data loading (should be ~instant)
# Increase workers if I/O bound

# Check GPU utilization
nvidia-smi dmon -s u
```

### NaN Loss
```bash
# Reduce learning rate
--learning-rate 1e-4

# Increase warmup
--warmup-steps 2000

# Check gradient clipping
--grad-clip 0.5
```

### Checkpoint Issues
```bash
# Verify checkpoint integrity
python3 -c "
import torch
ckpt = torch.load('checkpoints/step_1000.pt')
print(f'Step: {ckpt[\"step\"]}')
print(f'Tokens: {ckpt[\"tokens_seen\"]/1e9:.2f}B')
"
```

## File Structure

```
picotalk/
├── models/
│   ├── model.py           # Transformer implementation
│   └── config.py          # Model configurations
├── train/
│   ├── train.py           # Main training script
│   ├── data_loader.py     # Data loading utilities
│   └── scaling_experiments.py
├── eval/
│   └── evaluate.py        # Evaluation script
├── data/
│   ├── download_datasets.py
│   ├── tokenize_data.py
│   ├── raw/               # Downloaded .jsonl files
│   └── tokenized/         # Tokenized .bin files
├── configs/
│   └── data_config.yaml   # Data mix configuration
├── checkpoints/           # Model checkpoints
└── TRAINING.md           # This file
```

## Citation

If you use Picotalk, please cite:

```bibtex
@software{picotalk2024,
  title = {Picotalk: A 1B Parameter Language Model},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/picotalk}
}
```
