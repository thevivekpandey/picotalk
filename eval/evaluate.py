#!/usr/bin/env python3
"""
Evaluation script for Picotalk.

Supports:
- Perplexity on custom validation sets
- Zero-shot evaluation on common benchmarks (HellaSwag, PIQA, etc.)
- Text generation samples
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.config import ModelConfig, get_config
from models.model import Transformer
from transformers import AutoTokenizer


class Evaluator:
    """Model evaluation class."""

    def __init__(
        self,
        model: Transformer,
        tokenizer,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def compute_perplexity(
        self,
        texts: List[str],
        batch_size: int = 4,
        max_length: int = 2048,
    ) -> float:
        """Compute perplexity on a list of texts."""
        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Create targets (shift by 1)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = -1  # Ignore last token

            # Mask out padding
            targets[attention_mask == 0] = -1

            # Forward pass
            logits, loss = self.model(input_ids, targets)

            # Count valid tokens
            valid_tokens = (targets != -1).sum().item()

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        return perplexity

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> str:
        """Generate text from a prompt."""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    @torch.no_grad()
    def eval_multiple_choice(
        self,
        context: str,
        choices: List[str],
    ) -> int:
        """
        Evaluate multiple choice question.
        Returns index of most likely choice.
        """
        likelihoods = []

        for choice in choices:
            # Combine context and choice
            text = context + " " + choice

            # Tokenize
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

            # Get logits
            logits, _ = self.model(input_ids)

            # Compute likelihood of the choice tokens
            choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            choice_start = len(self.tokenizer.encode(context, add_special_tokens=False))

            choice_logits = logits[0, choice_start:choice_start + len(choice_ids)]
            choice_targets = torch.tensor(choice_ids, device=self.device)

            # Compute log probability
            log_probs = F.log_softmax(choice_logits, dim=-1)
            choice_log_probs = log_probs.gather(
                1, choice_targets.unsqueeze(1)
            ).squeeze(1)

            likelihood = choice_log_probs.sum().item()
            likelihoods.append(likelihood)

        return likelihoods.index(max(likelihoods))


def load_model(checkpoint_path: Path, device: str = 'cuda') -> tuple:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    model_config_dict = checkpoint['model_config']
    model_config = ModelConfig(**model_config_dict)

    # Build model
    model = Transformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    print(f"Model loaded: {model.get_num_params()/1e6:.1f}M parameters")
    print(f"Training step: {checkpoint['step']}")
    print(f"Tokens seen: {checkpoint['tokens_seen']/1e9:.2f}B")

    return model, tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate Picotalk")

    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    # Evaluation tasks
    parser.add_argument('--perplexity', type=str, default=None,
                        help='Compute perplexity on text file (one doc per line)')
    parser.add_argument('--generate', action='store_true', help='Generate sample texts')
    parser.add_argument('--prompts', type=str, default=None,
                        help='File with prompts for generation (one per line)')

    # Generation options
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')

    args = parser.parse_args()

    # Load model
    model, tokenizer, checkpoint = load_model(Path(args.checkpoint), args.device)
    evaluator = Evaluator(model, tokenizer, args.device)

    print(f"\n{'='*60}")
    print("Evaluation")
    print(f"{'='*60}\n")

    # Perplexity evaluation
    if args.perplexity:
        print(f"Computing perplexity on: {args.perplexity}")
        with open(args.perplexity, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]

        ppl = evaluator.compute_perplexity(texts)
        print(f"\nPerplexity: {ppl:.2f}")

    # Text generation
    if args.generate:
        if args.prompts:
            # Load prompts from file
            with open(args.prompts, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default prompts
            prompts = [
                "Once upon a time",
                "The meaning of life is",
                "In Python, you can",
                "The theory of relativity states that",
                "def fibonacci(n):",
            ]

        print("\nGenerating text samples:\n")
        for i, prompt in enumerate(prompts, 1):
            print(f"{'='*60}")
            print(f"Prompt {i}: {prompt}")
            print(f"{'-'*60}")

            generated = evaluator.generate_text(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            print(generated)
            print()


if __name__ == '__main__':
    main()
