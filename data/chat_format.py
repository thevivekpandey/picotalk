"""
Chat formatting for Picotalk SFT.

Uses the Mistral instruct format, which needs NO new special tokens
(important: our vocab is frozen at 32000, so we cannot add <|im_start|> etc.
without resizing embeddings):

    <s>[INST] {user} [/INST] {assistant}</s>[INST] {user2} [/INST] {assistant2}</s>

- A system prompt, if present, is prepended to the first user message.
- BOS (<s>) starts each conversation; EOS (</s>) ends each assistant turn,
  so the model learns to stop generating naturally.
- Loss masks: 1 on assistant response tokens + their EOS (trained),
  0 on everything else (prompt tokens, not trained).

Token/mask pairs plug directly into the model's loss: targets with mask==0
are set to -1, which matches ignore_index=-1 in Transformer.forward.
"""

from typing import List, Dict, Tuple


def _merge_system(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Fold a leading system message into the first user message."""
    if messages and messages[0]["role"] == "system":
        system = messages[0]["content"].strip()
        rest = messages[1:]
        if system and rest and rest[0]["role"] == "user":
            rest = [{"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}] + rest[1:]
        return rest
    return messages


def format_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Format messages into a generation prompt string (ends right where the
    assistant should start writing). Used at inference/sampling time.

    Note: does NOT include the leading <s> - the tokenizer adds BOS itself
    when encoding with special tokens.
    """
    messages = _merge_system(messages)

    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"[INST] {msg['content'].strip()} [/INST]")
        elif msg["role"] == "assistant":
            # Completed earlier turns (multi-turn context)
            parts.append(f" {msg['content'].strip()}</s>")

    return "".join(parts)


def tokenize_conversation(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int = None,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a conversation into (token_ids, loss_mask).

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        tokenizer: HF tokenizer (Mistral)
        max_length: optional truncation length

    Returns:
        ids:  list of token ids, starting with BOS
        mask: parallel list, 1 = train on this token (assistant + EOS), 0 = don't
    """
    messages = _merge_system(messages)

    ids: List[int] = [tokenizer.bos_token_id]
    mask: List[int] = [0]

    for msg in messages:
        content = msg["content"].strip()

        if msg["role"] == "user":
            # Prompt part: not trained
            piece = tokenizer.encode(f"[INST] {content} [/INST]", add_special_tokens=False)
            ids.extend(piece)
            mask.extend([0] * len(piece))

        elif msg["role"] == "assistant":
            # Response part: trained, including the EOS so the model learns to stop
            piece = tokenizer.encode(content, add_special_tokens=False)
            ids.extend(piece)
            mask.extend([1] * len(piece))
            ids.append(tokenizer.eos_token_id)
            mask.append(1)

        # Any other role (e.g., unknown) is skipped silently by design:
        # converters upstream normalize roles before calling this.

    if max_length is not None and len(ids) > max_length:
        ids = ids[:max_length]
        mask = mask[:max_length]

    return ids, mask
