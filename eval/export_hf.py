#!/usr/bin/env python3
"""
Export a Picotalk checkpoint to a HuggingFace `LlamaForCausalLM` directory.

Picotalk IS a Llama (RoPE, RMSNorm, SwiGLU, GQA, no biases, tied embeddings),
so the export is a key rename plus one real transformation: the RoPE weight
layout. Picotalk rotates adjacent dimension pairs (0,1), (2,3), ... via
`view_as_complex`; HF Llama rotates (i, i + head_dim/2) via `rotate_half`.
Permuting the rows of q_proj / k_proj per head makes the two identical.

Once exported, the model runs under vLLM, transformers and lm-eval-harness:

    python3 eval/export_hf.py checkpoints/sft_1b/final.pt hf/sft_1b --verify

The tokenizer is saved alongside with a chat template that reproduces
data/chat_format.tokenize_conversation() token-for-token, so
`--apply_chat_template` in lm-eval gives the SFT model exactly the prompt
shape it was trained on.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.model import Transformer

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"

# Mirrors data/chat_format.py: BOS once, "[INST] user [/INST]" untrained,
# " assistant</s>" trained, leading system message folded into the first user
# turn with a blank line. The generation prompt ends at "[/INST]".
#
# SFT encoded each segment separately, so every "[INST]" and every reply got
# SentencePiece's word-initial "▁" (token 733 "▁[", not 28792 "["). When the
# rendered string is tokenized in one go that "▁" only appears where there is
# an actual space, hence the explicit space before each "[INST]" and reply.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system = messages[0]['content'] | trim %}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set system = '' %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{% if loop.index0 == 0 and system %}"
    " [INST] {{ (system ~ '\n\n' ~ message['content']) | trim }} [/INST]"
    "{% else %}"
    " [INST] {{ message['content'] | trim }} [/INST]"
    "{% endif %}"
    "{% elif message['role'] == 'assistant' %}"
    " {{ message['content'] | trim }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


def load_native(ckpt_path: str):
    """Return (ModelConfig, state_dict, metadata) from a Picotalk checkpoint.

    mmap=True keeps the optimizer state (2/3 of the file) on disk.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)

    valid = set(inspect.signature(ModelConfig.__init__).parameters)
    cfg = ModelConfig(**{k: v for k, v in ckpt["model_config"].items() if k in valid})

    sd = {}
    for k, v in ckpt["model_state_dict"].items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        if k == "freqs_cis":  # derived from config; older checkpoints persisted it
            continue
        sd[k] = v

    meta = {k: ckpt[k] for k in ("step", "tokens_seen") if k in ckpt}
    return cfg, sd, meta


def permute_rope(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Interleaved-pair RoPE layout -> rotate_half layout, per head.

    Row 2j of each head becomes row j, row 2j+1 becomes row j + head_dim/2.
    """
    out_dim, in_dim = w.shape
    assert out_dim == n_heads * head_dim, (w.shape, n_heads, head_dim)
    return (
        w.view(n_heads, head_dim // 2, 2, in_dim)
        .transpose(1, 2)
        .reshape(out_dim, in_dim)
    )


def to_hf_state_dict(cfg: ModelConfig, sd: dict) -> dict:
    head_dim = cfg.n_embd // cfg.n_head
    hf = {"model.embed_tokens.weight": sd["tok_embeddings.weight"]}

    assert torch.equal(sd["output.weight"], sd["tok_embeddings.weight"]), \
        "output.weight is not tied to tok_embeddings.weight; export assumes tying"

    for i in range(cfg.n_layer):
        src = f"layers.{i}."
        dst = f"model.layers.{i}."
        hf[dst + "self_attn.q_proj.weight"] = permute_rope(sd[src + "attention.wq.weight"], cfg.n_head, head_dim)
        hf[dst + "self_attn.k_proj.weight"] = permute_rope(sd[src + "attention.wk.weight"], cfg.n_kv_head, head_dim)
        hf[dst + "self_attn.v_proj.weight"] = sd[src + "attention.wv.weight"]
        hf[dst + "self_attn.o_proj.weight"] = sd[src + "attention.wo.weight"]
        hf[dst + "mlp.gate_proj.weight"] = sd[src + "feed_forward.w1.weight"]
        hf[dst + "mlp.down_proj.weight"] = sd[src + "feed_forward.w2.weight"]
        hf[dst + "mlp.up_proj.weight"] = sd[src + "feed_forward.w3.weight"]
        hf[dst + "input_layernorm.weight"] = sd[src + "attention_norm.weight"]
        hf[dst + "post_attention_layernorm.weight"] = sd[src + "ffn_norm.weight"]

    hf["model.norm.weight"] = sd["norm.weight"]

    expected = {"tok_embeddings.weight", "output.weight", "norm.weight"} | {
        f"layers.{i}.{name}"
        for i in range(cfg.n_layer)
        for name in (
            "attention.wq.weight", "attention.wk.weight", "attention.wv.weight",
            "attention.wo.weight", "feed_forward.w1.weight", "feed_forward.w2.weight",
            "feed_forward.w3.weight", "attention_norm.weight", "ffn_norm.weight",
        )
    }
    unused = set(sd) - expected
    assert not unused, f"unconverted checkpoint keys: {sorted(unused)}"
    return hf


def build_hf_config(cfg: ModelConfig):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.n_embd,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=cfg.n_head,
        num_key_value_heads=cfg.n_kv_head,
        head_dim=cfg.n_embd // cfg.n_head,
        max_position_embeddings=cfg.block_size,
        rms_norm_eps=1e-6,  # RMSNorm default in models/model.py
        rope_theta=cfg.rope_theta,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )


def export(ckpt_path: str, out_dir: str, dtype: torch.dtype):
    from transformers import AutoTokenizer, LlamaForCausalLM

    cfg, sd, meta = load_native(ckpt_path)
    print(f"Checkpoint: {ckpt_path}")
    print(f"  {cfg.n_layer}L x {cfg.n_embd}d, {cfg.n_head} heads / {cfg.n_kv_head} kv, "
          f"ffn {cfg.intermediate_size}, ctx {cfg.block_size}, vocab {cfg.vocab_size}")
    if meta:
        print(f"  {meta}")

    hf_config = build_hf_config(cfg)
    model = LlamaForCausalLM(hf_config)
    missing, unexpected = model.load_state_dict(to_hf_state_dict(cfg, sd), strict=False)
    # lm_head is tied to embed_tokens and is the only key allowed to be missing.
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing and not unexpected, (missing, unexpected)
    model.tie_weights()
    model = model.to(dtype)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(out)

    with open(out / "picotalk_export.json", "w") as f:
        json.dump({"source_checkpoint": str(ckpt_path), "dtype": str(dtype), **meta}, f, indent=2)

    n_bytes = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    print(f"Exported to {out} ({n_bytes / 1e9:.2f} GB weights, {dtype})")
    return cfg, sd


@torch.no_grad()
def verify(cfg: ModelConfig, sd: dict, out_dir: str, dtype: torch.dtype, seq_len: int = 64):
    """Check the export is the same function as the native model.

    Both sides run in fp32 on CPU. The native weights are first rounded through
    the export dtype so the comparison isolates the conversion (key mapping and
    RoPE permutation) from bf16 rounding, which vLLM would apply either way.
    """
    from transformers import AutoTokenizer, LlamaForCausalLM

    from data.chat_format import tokenize_conversation

    native = Transformer(cfg)
    native.load_state_dict({k: v.to(dtype).float() for k, v in sd.items()})
    native.eval()

    hf = LlamaForCausalLM.from_pretrained(out_dir, dtype=torch.float32)
    hf.eval()

    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, seq_len), generator=g)
    ids[0, 0] = tokenizer.bos_token_id

    native_logits, _ = native(ids)
    hf_logits = hf(input_ids=ids).logits

    max_diff = (native_logits - hf_logits).abs().max().item()
    scale = native_logits.abs().max().item()
    argmax_agree = (native_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    print(f"Logit check over {ids.numel()} positions: max |diff| = {max_diff:.3e} "
          f"(logit scale {scale:.1f}), argmax agreement = {argmax_agree * 100:.2f}%")
    assert max_diff < 1e-3 * max(scale, 1.0), "exported model does not match native model"

    # The chat template must reproduce SFT tokenization exactly.
    convs = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "system", "content": "Be brief."},
         {"role": "user", "content": "  Name a colour. "},
         {"role": "assistant", "content": "Red."},
         {"role": "user", "content": "Another?"}],
    ]
    for messages in convs:
        want, _ = tokenize_conversation(messages, tokenizer)
        got = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(got, "input_ids"):  # transformers >= 5 returns a BatchEncoding
            got = got["input_ids"]
        assert list(got) == want, f"chat template mismatch:\n  template: {got}\n  sft:      {want}"
    print("Chat template matches tokenize_conversation() on "
          f"{len(convs)} conversations (incl. system + multi-turn).")
    print("VERIFY OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Picotalk .pt checkpoint")
    parser.add_argument("out_dir", help="Output HF model directory")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="Saved weight dtype (default bfloat16, which is what "
                             "training's autocast computed in)")
    parser.add_argument("--verify", action="store_true",
                        help="Reload the export and check logits and chat tokenization against the native model (CPU)")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    cfg, sd = export(args.checkpoint, args.out_dir, dtype)
    if args.verify:
        verify(cfg, sd, args.out_dir, dtype)


if __name__ == "__main__":
    main()
