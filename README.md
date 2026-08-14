# Picotalk

A compact 1B parameter language model trained from scratch within a $2000 budget.

**Design Philosophy**: Complete, reproducible LLM pipeline from tokenization to deployment, optimized for limited compute (500 H100-hours).

## Quick Start

```bash
# Test setup
python3 test_setup.py

# Download data (30B tokens, ~150GB)
python3 data/download_datasets.py --datasets all

# Tokenize
python3 data/tokenize_data.py

# Train (single GPU)
python3 train/train.py --model 1b --wandb

# Evaluate
python3 eval/evaluate.py ./checkpoints/best.pt --generate
```

## Features

### Architecture
- **1.07B parameters** (30 layers, 1792 dim, 16 heads)
- **Llama-style**: RoPE, RMSNorm, SwiGLU, GQA (4 KV heads)
- **2048 context length**
- **Mistral tokenizer** (32k vocab)

### Training Data (30B tokens)
- **DCLM** (10B): High-quality web text
- **Cosmopedia** (5B): Synthetic educational content
- **FineWeb-Edu** (6B): Educational web pages
- **StarCoder Python** (6B): Python code
- **OpenWebMath** (2B): Mathematical content
- **ArXiv** (1B): Scientific papers

### Training Features
- **Mixed precision** (bfloat16/float16)
- **Gradient accumulation** for large effective batches
- **Distributed training** (DDP support)
- **Cosine LR schedule** with warmup
- **Wandb integration** for monitoring
- **Checkpointing** and resumption

## Project Structure

```
picotalk/
├── models/
│   ├── model.py              # Transformer implementation
│   └── config.py             # Model configurations
├── train/
│   ├── train.py              # Main training script
│   ├── data_loader.py        # Memory-mapped data loading
│   └── scaling_experiments.py # Scaling law experiments
├── eval/
│   └── evaluate.py           # Evaluation and generation
├── data/
│   ├── download_datasets.py  # Data download
│   ├── tokenize_data.py      # Tokenization
│   ├── raw/                  # Downloaded .jsonl files
│   └── tokenized/            # Tokenized .bin files
├── configs/
│   └── data_config.yaml      # Data mix configuration
├── test_setup.py             # Setup verification
├── README.md                 # This file
└── TRAINING.md               # Detailed training guide
```

## Documentation

- **[TRAINING.md](TRAINING.md)**: Complete training guide
  - Data download and tokenization
  - Scaling experiments
  - Training configurations
  - Multi-GPU setup
  - Evaluation and troubleshooting

## Model Configurations

| Config | Params | Layers | Dim | Heads | KV Heads | Use Case |
|--------|--------|--------|-----|-------|----------|----------|
| test   | 0.5M   | 2      | 128 | 2     | 2        | Quick testing |
| 50m    | 50M    | 6      | 512 | 8     | 4        | Scaling experiments |
| 100m   | 100M   | 8      | 768 | 8     | 4        | Scaling experiments |
| 200m   | 200M   | 12     | 768 | 12    | 4        | Scaling experiments |
| 350m   | 350M   | 16     | 1024| 16    | 4        | Scaling experiments |
| **1b** | **1.07B** | **30** | **1792** | **16** | **4** | **Final model** |

## Training Examples

### Scaling Experiments

```bash
# List available experiments
python3 train/scaling_experiments.py --list

# Run small experiment (~15 min on H100)
python3 train/scaling_experiments.py --experiments 50m_1b

# Run multiple experiments
python3 train/scaling_experiments.py --experiments 100m_3b 200m_5b
```

### Single GPU Training

```bash
python3 train/train.py \
  --model 1b \
  --batch-size 8 \
  --grad-accum-steps 16 \
  --max-steps 30000 \
  --learning-rate 3e-4 \
  --wandb \
  --run-name picotalk-1b
```

### Multi-GPU Training (4x H100)

```bash
torchrun --nproc_per_node=4 train/train.py \
  --model 1b \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --max-steps 30000 \
  --ddp \
  --wandb \
  --run-name picotalk-1b-4gpu
```

### Resume Training

```bash
python3 train/train.py \
  --model 1b \
  --resume ./checkpoints/step_10000.pt
```

## Evaluation

### Text Generation

```bash
# Generate with default prompts
python3 eval/evaluate.py ./checkpoints/best.pt --generate

# Custom prompts
python3 eval/evaluate.py ./checkpoints/best.pt \
  --generate \
  --prompts ./my_prompts.txt \
  --temperature 0.7 \
  --max-tokens 200
```

### Perplexity

```bash
python3 eval/evaluate.py ./checkpoints/best.pt \
  --perplexity ./validation_data.txt
```

## Compute Budget

**Total**: $2000 (~500 H100-hours @ $4/hour)

- **Scaling experiments**: ~50 hours
- **Final 1B training**: ~400 hours
  - Option A: 1x H100 × 400 hours (~17 days)
  - Option B: 4x H100 × 100 hours (~4 days)
  - Option C: 8x H100 × 50 hours (~2 days)
- **Reserve**: ~50 hours (fine-tuning, RL)

## Expected Performance

**1B model @ 30B tokens**:
- Validation perplexity: ~15-20 (estimated)
- Coherent text generation
- Basic reasoning capabilities
- Python code generation (simple functions)
- Instruction following (after SFT)

**Comparison**:
- Similar to SmolLM (1.7B) or early TinyLlama checkpoints
- Will benefit from SFT and RL fine-tuning

## Requirements

```bash
pip install torch transformers datasets wandb tqdm pyyaml psutil
```

**Optional**:
- `zstandard` (for .jsonl.zst compression)
- `wandb` (for experiment tracking)

## Development Workflow

1. **Setup**: Run `python3 test_setup.py` to verify installation
2. **Data**: Download and tokenize datasets
3. **Scaling**: Run small experiments to derive scaling laws
4. **Training**: Train final 1B model
5. **Evaluation**: Generate samples and measure perplexity
6. **Fine-tuning**: SFT and RL (future work)

## Next Steps

After pretraining:

1. **Supervised Fine-Tuning (SFT)**
   - Instruction datasets (Alpaca, ShareGPT, etc.)
   - Chat formatting
   - ~5B tokens, 2-3 epochs

2. **Reinforcement Learning**
   - DPO/RLHF on preference data
   - Human evaluation
   - ~1B tokens

3. **Deployment**
   - Quantization (int8/int4)
   - GGUF export for llama.cpp
   - Inference optimization

## Citation

```bibtex
@software{picotalk2024,
  title = {Picotalk: A 1B Parameter Language Model},
  author = {Vivek Pandey},
  year = {2024},
  url = {https://github.com/yourusername/picotalk}
}
```

## License

MIT License

## Acknowledgments

- Architecture inspired by Llama 2/3 and Mistral
- Training approach based on scaling laws (Kaplan et al., Hoffmann et al.)
- Data curation follows FineWeb and DCLM principles

---

**Status**: ✅ Model and training code complete | ⏳ Awaiting full dataset download
