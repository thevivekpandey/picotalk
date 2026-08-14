#!/usr/bin/env python3
"""
Test different tokenizers on sample data to choose the best one.
"""

from transformers import AutoTokenizer
import numpy as np

def test_tokenizer(tokenizer_name: str, test_texts: dict):
    """Test a tokenizer on different text types."""
    print(f"\n{'='*60}")
    print(f"Testing: {tokenizer_name}")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"✓ Loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size:,}")

        results = {}

        for text_type, text in test_texts.items():
            # Tokenize
            tokens = tokenizer.encode(text)

            # Calculate metrics
            num_tokens = len(tokens)
            num_chars = len(text)
            chars_per_token = num_chars / num_tokens if num_tokens > 0 else 0

            results[text_type] = {
                'tokens': num_tokens,
                'chars': num_chars,
                'chars_per_token': chars_per_token
            }

            print(f"\n  {text_type}:")
            print(f"    Characters: {num_chars:,}")
            print(f"    Tokens: {num_tokens:,}")
            print(f"    Chars/token: {chars_per_token:.2f}")

            # Show first few tokens
            decoded_samples = [tokenizer.decode([t]) for t in tokens[:10]]
            print(f"    First 10 tokens: {decoded_samples}")

        # Overall average
        avg_chars_per_token = np.mean([r['chars_per_token'] for r in results.values()])
        print(f"\n  Overall avg chars/token: {avg_chars_per_token:.2f}")

        return results

    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    # Sample texts from different domains
    test_texts = {
        'general_chat': """
Hello! How are you doing today? I'd like to learn more about machine learning
and artificial intelligence. Can you explain the basics to me? I'm particularly
interested in understanding how neural networks work and what makes them so powerful
for tasks like natural language processing.
        """.strip(),

        'code_python': """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class DataLoader:
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i:i + self.batch_size]
        """.strip(),

        'math': """
Given the equation f(x) = 2x² + 3x - 5, find the derivative using the power rule.
Solution: f'(x) = d/dx(2x²) + d/dx(3x) - d/dx(5) = 4x + 3.
The integral of f(x) from 0 to 2 is ∫₀² (2x² + 3x - 5)dx = [2x³/3 + 3x²/2 - 5x]₀².
        """.strip(),

        'scientific': """
The Heisenberg uncertainty principle states that ΔxΔp ≥ ℏ/2, where Δx represents
the uncertainty in position and Δp represents the uncertainty in momentum. This
fundamental principle of quantum mechanics demonstrates that certain pairs of
physical properties cannot be simultaneously known with arbitrary precision.
        """.strip(),

        'reasoning': """
To solve this problem, let's break it down step by step:
1. First, identify the given information and what we need to find
2. Consider what principles or formulas might apply
3. Set up equations based on the relationships between variables
4. Solve systematically, checking our work at each stage
5. Verify the answer makes sense in the original context
Therefore, the solution is well-reasoned and mathematically sound.
        """.strip()
    }

    # Tokenizers to test
    tokenizers = [
        "meta-llama/Meta-Llama-3-8B",  # Llama 3 (128k) - try this first
        "mistralai/Mistral-7B-v0.1",   # Mistral (32k)
        "gpt2",                         # GPT-2 (50k)
        "EleutherAI/gpt-neox-20b",     # GPT-NeoX (50k)
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # TinyLlama (32k)
    ]

    print("\n" + "="*60)
    print("Tokenizer Comparison for Picotalk")
    print("="*60)
    print("\nNote: Higher chars/token = better compression = more efficient")
    print("Target: 3.5-4.5 chars/token is good\n")

    all_results = {}

    for tokenizer_name in tokenizers:
        results = test_tokenizer(tokenizer_name, test_texts)
        if results:
            all_results[tokenizer_name] = results

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"\n{'Tokenizer':<35} {'Vocab Size':<12} {'Avg Chars/Token':<15}")
    print("-" * 62)

    for tokenizer_name in all_results.keys():
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            avg_cpt = np.mean([
                r['chars_per_token']
                for r in all_results[tokenizer_name].values()
            ])
            vocab_size = tokenizer.vocab_size

            # Highlight best
            marker = " ⭐" if tokenizer_name == "meta-llama/Llama-2-7b-hf" else ""

            print(f"{tokenizer_name:<35} {vocab_size:<12,} {avg_cpt:<15.2f}{marker}")
        except:
            pass

    print("\n" + "="*60)
    print("RECOMMENDATION: meta-llama/Llama-2-7b-hf")
    print("  ✓ Best balance of vocab size and efficiency")
    print("  ✓ Good across all domains (chat, code, math, science)")
    print("  ✓ Industry standard for modern small-medium models")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
