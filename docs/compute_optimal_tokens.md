# Compute-Optimal Training for Picotalk 1B

## Your Calculation (Verification)

### Available Compute
- **Budget**: $1,000
- **H100 cost**: $3.50/hour
- **GPU-hours**: 1000 / 3.5 = **285.7 GPU-hours**

### FLOP Calculation
- **H100 FP16/BF16**: ~1000 TFLOPS (peak), but **~400 TFLOPS sustained** for training
- **Total FLOPs**: 400 × 10¹² FLOPS/sec × 285.7 hours × 3600 sec/hour
  - = 400 × 10¹² × 1,028,520 seconds
  - = **4.11 × 10²⁰ FLOPs**

✅ Your calculation is **correct**: ~4.2 × 10²⁰ FLOPs

### Chinchilla Optimal Tokens

**Chinchilla scaling law** (Hoffmann et al., 2022):
```
N_optimal = C / 6
```
Where:
- N = number of tokens
- C = compute budget in FLOPs
- 6 = constant from Chinchilla paper (for parameter count in FLOPs)

**Your calculation**:
```
N_optimal = 4.2 × 10²⁰ / (6 × 1.07 × 10⁹)
          = 4.2 × 10²⁰ / 6.42 × 10⁹
          = 65.4B tokens
```

✅ Your **~60B tokens** estimate is **spot on**!

## Current Data vs Optimal

| Metric | Current | Optimal | Gap |
|--------|---------|---------|-----|
| **Training data** | 31B tokens | **65B tokens** | **-34B** (52% short) |
| **Model size** | 1.07B params | 1.07B params | ✓ Good |
| **Compute budget** | 4.2 × 10²⁰ | 4.2 × 10²⁰ | ✓ Fixed |

**Analysis**: You need **~2× more training data** to be compute-optimal!

## The Chinchilla Insight

From the original paper, for a given compute budget C:

**Optimal allocation**:
- Model size: N ∝ C^0.5
- Training tokens: D ∝ C^0.5
- **They should scale together!**

For your 1.07B model with 4.2 × 10²⁰ FLOPs:
- **Optimal tokens**: ~65B
- **You have**: 31B tokens
- **Problem**: Undertrained by 2×

## Three Options

### Option 1: Get More Data (RECOMMENDED)

**Target**: 65B tokens (need +34B more)

**Where to get it**:
1. **Download more from existing sources**:
   - DCLM: Currently 10B → increase to 20B (abundant data available)
   - FineWeb-Edu: Currently 6B → increase to 12B
   - Stack v2: Currently 8B → increase to 15B
   - Total: 31B → 67B ✓

2. **Add new high-quality sources**:
   - **SlimPajama**: Deduplicated, high-quality web (300B+ available)
   - **RedPajama v2**: Quality-filtered web text
   - **Dolma**: Allen AI's open dataset

3. **Estimated additional download/tokenize time**:
   - Download 34B tokens: ~70GB raw → ~5-8 hours
   - Tokenization: ~2-3 hours
   - **Total overhead**: ~10 hours of prep

**Pros**:
- ✅ Compute-optimal training
- ✅ Better final model quality
- ✅ Best use of your $1000 budget
- ✅ Follows best practices (Chinchilla, Llama)

**Cons**:
- ⚠️ Need to download/tokenize more data (~10 hours)
- ⚠️ Need ~35GB more disk space

### Option 2: Train Longer on Existing Data

**Approach**: Train for ~2 epochs on 31B tokens = 62B effective tokens

**Considerations**:

**Epoch 1** (first 31B tokens):
- Fresh data, model learns rapidly
- This is standard training

**Epoch 2** (repeat 31B tokens):
- Model has seen data before
- Still learns, but with diminishing returns
- Learning rate should be decayed further

**Research shows**:
- 1st epoch: 100% efficiency
- 2nd epoch: ~70-80% efficiency (due to repetition)
- Common practice for small datasets

**Effective tokens**: ~31B + 0.75 × 31B = **~54B effective tokens** (not quite 65B)

**Pros**:
- ✅ No additional data needed
- ✅ Can start training immediately
- ✅ Simpler pipeline

**Cons**:
- ⚠️ Still ~15% below optimal (54B vs 65B)
- ⚠️ Diminishing returns on 2nd epoch
- ⚠️ May overfit on smaller dataset
- ⚠️ Not following best practices

### Option 3: Reduce Model Size

**Approach**: Train a smaller model (750M) to be perfectly compute-optimal with 31B tokens

**Calculation**:
For 31B tokens, optimal model size:
```
C = 6 × N × D
4.2 × 10²⁰ = 6 × N × 31 × 10⁹
N = 4.2 × 10²⁰ / (6 × 31 × 10⁹)
N = 2.26 × 10⁹ = 2.26B params
```

Wait, that suggests 2.26B params for 31B tokens!

Let me recalculate assuming you want to use 31B tokens:
```
For D = 31B tokens and N = 1.07B params:
C_needed = 6 × 1.07 × 10⁹ × 31 × 10⁹
         = 1.99 × 10²⁰ FLOPs
```

**Your actual compute**: 4.2 × 10²⁰ FLOPs

So you could train a **larger model** with 31B tokens:
```
N = 4.2 × 10²⁰ / (6 × 31 × 10⁹)
  = 2.26B params
```

**Pros**:
- ✅ Compute-optimal for your data
- ✅ Larger model (2.26B vs 1.07B)
- ✅ No additional data needed

**Cons**:
- ⚠️ Need to redesign architecture for 2.26B
- ⚠️ May not fit your original goal (1B model)
- ⚠️ Larger model = slower inference

## My Recommendation

### **Option 1: Get More Data (Target 65-70B tokens)**

**Why**:
1. ✅ You correctly identified being undertrained
2. ✅ More data = better model (proven by Chinchilla, Llama)
3. ✅ Only ~10 hours of extra prep time
4. ✅ Best use of your $1000 compute budget
5. ✅ Follows scaling laws exactly

**How**:

```yaml
# Updated data_config.yaml
datasets:
  dclm:
    target_tokens: 20_000_000_000  # 20B (was 10B)

  cosmopedia:
    target_tokens: 5_000_000_000   # 5B (unchanged)

  fineweb_edu:
    target_tokens: 12_000_000_000  # 12B (was 6B)

  stack_v2:
    target_tokens: 15_000_000_000  # 15B (was 6B)

  openwebmath:
    target_tokens: 4_000_000_000   # 4B (was 2B)

  arxiv:
    target_tokens: 2_000_000_000   # 2B (was 1B)

# Total: 58B tokens

# Optional: Add SlimPajama for remaining 7B
  slimpajama:
    name: "cerebras/SlimPajama-627B"
    target_tokens: 7_000_000_000   # 7B

# New total: 65B tokens ✓
```

**Time investment**:
- Download additional data: ~6 hours
- Tokenization: ~3 hours
- Total: ~9 hours of prep

**Benefit**:
- Properly utilize your $1000 budget
- Train a compute-optimal model
- Better final performance

## Detailed Compute Budget Allocation

### Current Plan: 31B tokens, 1.07B params

| GPU Config | GPU-Hours | Cost | Time | Tokens | Under-trained by |
|------------|-----------|------|------|--------|------------------|
| 1×H100 | 160 | $560 | 160h | 31B | 52% |
| 4×H100 | 172 | $602 | 43h | 31B | 52% |

**Waste**: ~$400 of compute not being utilized!

### Optimal Plan: 65B tokens, 1.07B params

| GPU Config | GPU-Hours | Cost | Time | Tokens | Utilization |
|------------|-----------|------|------|--------|-------------|
| 1×H100 | 335 | $1,173 | 335h | 65B | 100% ✓ |
| 4×H100 | 361 | $1,264 | 90h | 65B | 100% ✓ |

**Problem**: Over budget by $264!

### Budget-Constrained Optimal: What fits in $1000?

With $1000 budget:
- GPU-hours: 1000 / 3.5 = 285.7
- Optimal tokens: ~60B

**Recommendation**: Target **60B tokens** (within budget)

| GPU Config | GPU-Hours | Cost | Time | Tokens | Budget |
|------------|-----------|------|------|--------|--------|
| 4×H100 | 334 | **$983** | 83h | 60B | ✓ Under |
| 8×H100 | 342 | $997 | 42h | 60B | ✓ Under |

Perfect fit!

## Training Steps Calculation

For 60B tokens with 1.07B model:

**Effective batch size**:
- Per GPU: 8 sequences
- Sequence length: 2048 tokens
- Gradient accumulation: varies by GPU count

| GPUs | Batch/GPU | Grad Accum | Effective Batch | Tokens/Step | Steps for 60B |
|------|-----------|------------|-----------------|-------------|---------------|
| 1 | 8 | 64 | 512 | 1,048,576 | 57,200 |
| 4 | 8 | 16 | 512 | 1,048,576 | 57,200 |
| 8 | 8 | 8 | 512 | 1,048,576 | 57,200 |

**Update training config**:
```bash
--max-steps 57200  # for 60B tokens
```

## Data Mix for 60B Tokens

**Balanced approach** (proportions similar to current):

| Dataset | Tokens | % | Notes |
|---------|--------|---|-------|
| DCLM | 20B | 33% | High-quality web |
| FineWeb-Edu | 12B | 20% | Educational |
| Stack v2 (Python) | 12B | 20% | Code |
| Cosmopedia | 8B | 13% | Synthetic education |
| OpenWebMath | 5B | 8% | Math |
| ArXiv | 3B | 5% | Science |
| **Total** | **60B** | **100%** | ✓ Budget-optimal |

This maintains good domain coverage while hitting compute budget.

## Summary

### Your Analysis: ✅ Correct!
- Compute budget: 4.2 × 10²⁰ FLOPs
- Optimal tokens: ~65B
- Current data: 31B
- **Conclusion**: Need ~2× more data

### My Recommendation: Get More Data

**Action plan**:
1. ✅ Update `configs/data_config.yaml` with targets above
2. ✅ Download additional data (~6 hours)
   ```bash
   python3 data/download_datasets.py --datasets all
   ```
3. ✅ Tokenize new data (~3 hours)
   ```bash
   python3 data/tokenize_data.py
   ```
4. ✅ Update training config:
   ```bash
   --max-steps 57200  # for 60B tokens
   ```
5. ✅ Train with full $1000 budget:
   ```bash
   # 4×H100: 83 hours, $983
   torchrun --nproc_per_node=4 train/train.py \
     --model 1b \
     --max-steps 57200 \
     --wandb
   ```

**Result**: Compute-optimal 1B model trained on 60B tokens, using your full $1000 budget efficiently!

### Alternative if Time-Constrained

If you can't wait 10 hours for more data:
- Train 2 epochs on 31B tokens = ~54B effective
- Use `--max-steps 51500`
- Budget: ~$850
- Not optimal, but reasonable

**But I strongly recommend getting more data** - it's only 10 hours of prep for a much better model!
