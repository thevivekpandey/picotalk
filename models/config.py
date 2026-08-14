"""
Model configuration for Picotalk.
Supports multiple sizes for scaling law experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for transformer models."""

    # Model size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # Architecture choices
    block_size: int = 2048  # Context length
    vocab_size: int = 32000  # Mistral tokenizer

    # Architecture style
    use_rope: bool = True  # RoPE (Llama-style) vs learned PE (GPT-style)
    use_rms_norm: bool = True  # RMSNorm (Llama) vs LayerNorm (GPT)
    use_swiglu: bool = True  # SwiGLU (Llama) vs GELU (GPT)
    use_gqa: bool = True  # Grouped Query Attention
    n_kv_head: Optional[int] = None  # For GQA, default = n_head (MHA)

    # FFN size
    intermediate_size: Optional[int] = None  # If None, = 4 * n_embd

    # Regularization
    dropout: float = 0.0  # Usually 0 for pretraining
    bias: bool = False  # Llama-style: no bias

    # Training
    rope_theta: float = 10000.0  # RoPE base frequency

    def __post_init__(self):
        """Set derived values."""
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd

        if self.n_kv_head is None:
            self.n_kv_head = self.n_head  # Default to MHA

        # Validate
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"


# Predefined configurations for scaling experiments
def get_config(name: str) -> ModelConfig:
    """Get predefined model configuration."""

    configs = {
        # Tiny models for testing
        "test": ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=128,
            block_size=256,
        ),

        # Scaling law experiment sizes
        "50m": ModelConfig(
            n_layer=6,
            n_head=8,
            n_embd=512,
            n_kv_head=4,  # GQA
        ),

        "100m": ModelConfig(
            n_layer=8,
            n_head=8,
            n_embd=768,
            n_kv_head=4,
        ),

        "200m": ModelConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
            n_kv_head=4,
        ),

        "350m": ModelConfig(
            n_layer=16,
            n_head=16,
            n_embd=1024,
            n_kv_head=4,
        ),

        "400m": ModelConfig(
            n_layer=18,
            n_head=16,
            n_embd=1024,
            n_kv_head=4,
        ),

        "500m": ModelConfig(
            n_layer=20,
            n_head=16,
            n_embd=1024,
            n_kv_head=4,
        ),

        "750m": ModelConfig(
            n_layer=24,
            n_head=16,
            n_embd=1280,
            n_kv_head=4,
        ),

        "800m": ModelConfig(
            n_layer=24,
            n_head=16,
            n_embd=1408,
            n_kv_head=4,
        ),

        "1b": ModelConfig(
            n_layer=30,
            n_head=16,
            n_embd=1792,
            n_kv_head=4,
            intermediate_size=4778,  # 2.67x (8/3) - Llama-style with SwiGLU
        ),

        # GPT-style configs (for comparison)
        "100m-gpt": ModelConfig(
            n_layer=8,
            n_head=8,
            n_embd=768,
            use_rope=False,
            use_rms_norm=False,
            use_swiglu=False,
            use_gqa=False,
            bias=True,
        ),

        "350m-gpt": ModelConfig(
            n_layer=16,
            n_head=16,
            n_embd=1024,
            use_rope=False,
            use_rms_norm=False,
            use_swiglu=False,
            use_gqa=False,
            bias=True,
        ),
    }

    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")

    return configs[name]


def count_parameters(config: ModelConfig) -> int:
    """Estimate parameter count for a config."""

    # Embedding
    params = config.vocab_size * config.n_embd  # Token embeddings

    if not config.use_rope:
        params += config.block_size * config.n_embd  # Position embeddings

    # Transformer blocks
    for _ in range(config.n_layer):
        # Attention
        # Q projections: n_head groups
        params += config.n_embd * (config.n_embd + (config.n_kv_head * 2 * config.n_embd // config.n_head))

        # Output projection
        params += config.n_embd * config.n_embd

        # MLP
        if config.use_swiglu:
            # SwiGLU has gate + up + down
            params += config.n_embd * config.intermediate_size  # gate
            params += config.n_embd * config.intermediate_size  # up
            params += config.intermediate_size * config.n_embd  # down
        else:
            # Standard FFN
            params += config.n_embd * config.intermediate_size  # up
            params += config.intermediate_size * config.n_embd  # down

        # Layer norms (or RMSNorm)
        if config.use_rms_norm:
            params += config.n_embd * 2  # Two RMSNorms per block (no bias)
        else:
            params += config.n_embd * 4  # Two LayerNorms with bias

        # Bias terms
        if config.bias:
            params += config.n_embd * 3  # Attention biases
            params += config.intermediate_size + config.n_embd  # MLP biases

    # Final layer norm
    if config.use_rms_norm:
        params += config.n_embd
    else:
        params += config.n_embd * 2

    # Output head (usually tied with embeddings, but count it)
    # params += config.vocab_size * config.n_embd  # Uncomment if not weight tied

    return params


if __name__ == "__main__":
    # Test configurations
    print("Picotalk Model Configurations\n")
    print(f"{'Config':<15} {'Params':<12} {'Layers':<8} {'Embd':<8} {'Heads':<8} {'KV Heads':<10}")
    print("-" * 70)

    for name in ["50m", "100m", "200m", "350m", "400m", "500m", "750m", "800m", "1b"]:
        cfg = get_config(name)
        params = count_parameters(cfg)
        print(f"{name:<15} {params/1e6:>10.1f}M {cfg.n_layer:<8} {cfg.n_embd:<8} {cfg.n_head:<8} {cfg.n_kv_head:<10}")

    print("\n" + "="*70)
    print("Architecture Features:")
    cfg = get_config("350m")
    print(f"  RoPE: {cfg.use_rope}")
    print(f"  RMSNorm: {cfg.use_rms_norm}")
    print(f"  SwiGLU: {cfg.use_swiglu}")
    print(f"  GQA: {cfg.use_gqa} (n_head={cfg.n_head}, n_kv_head={cfg.n_kv_head})")
    print(f"  Bias: {cfg.bias}")
    print(f"  Context: {cfg.block_size}")
    print(f"  Vocab: {cfg.vocab_size}")
