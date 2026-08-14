"""
Picotalk: Llama-style transformer model.

Features:
- RoPE (Rotary Position Embeddings)
- RMSNorm
- SwiGLU activation
- Grouped Query Attention (GQA)
- No bias terms
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials for RoPE.

    Returns:
        freqs_cis: complex tensor of shape (end, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex exponentials
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: query tensor (batch, seqlen, n_heads, head_dim)
        xk: key tensor (batch, seqlen, n_kv_heads, head_dim)
        freqs_cis: precomputed frequencies (seqlen, head_dim//2)
    """
    # Reshape to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis to match shapes
    freqs_cis = freqs_cis[:xq_.shape[1], :]  # (seqlen, head_dim//2)
    freqs_cis = freqs_cis[None, :, None, :]  # (1, seqlen, 1, head_dim//2)

    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped Query Attention.

    Args:
        x: (batch, seqlen, n_kv_heads, head_dim)
        n_rep: number of times to repeat each KV head

    Returns:
        (batch, seqlen, n_kv_heads * n_rep, head_dim)
    """
    if n_rep == 1:
        return x

    batch, seqlen, n_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(batch, seqlen, n_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seqlen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head  # repetition factor for GQA
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V projections (GQA uses fewer KV heads)
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        # QKV projections
        xq = self.wq(x).view(batch, seqlen, self.n_head, self.head_dim)
        xk = self.wk(x).view(batch, seqlen, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(batch, seqlen, self.n_kv_head, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat KV heads for GQA
        xk = repeat_kv(xk, self.n_rep)  # (batch, seqlen, n_head, head_dim)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose for attention: (batch, n_head, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Combine with values
        output = torch.matmul(attn, xv)  # (batch, n_head, seqlen, head_dim)

        # Reshape and project out
        output = output.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.intermediate_size

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)  # down
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)  # up
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate) * up, then down
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        out = self.w2(gate * up)

        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        # Pre-norm FFN with residual
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    """Llama-style transformer language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final norm and output projection
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (share embeddings with output)
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size,
            config.rope_theta,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights (Llama-style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: (batch, seqlen) input token IDs
            targets: (batch, seqlen) target token IDs (for loss computation)

        Returns:
            logits: (batch, seqlen, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        batch, seqlen = tokens.shape
        device = tokens.device

        # Token embeddings
        h = self.tok_embeddings(tokens)

        # Move freqs_cis to device
        freqs_cis = self.freqs_cis[:seqlen].to(device)

        # Create causal mask
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask[None, None, :, :]  # (1, 1, seqlen, seqlen)

        # Transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # Final norm and output projection
        h = self.norm(h)
        logits = self.output(h)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_num_params(self, non_embedding=True):
        """Return number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: (batch, seqlen) starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k tokens

        Returns:
            (batch, seqlen + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def build_model(config: ModelConfig) -> Transformer:
    """Build a transformer model from config."""
    model = Transformer(config)
    print(f"Model built: {model.get_num_params()/1e6:.1f}M parameters")
    return model
