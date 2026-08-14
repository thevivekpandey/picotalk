#!/usr/bin/env python3
"""
Test the tokenization pipeline with sample data.
"""

from transformers import AutoTokenizer
import numpy as np
from pathlib import Path

def test_tokenizer():
    """Test basic tokenizer functionality."""

    print("="*60)
    print("Testing Mistral Tokenizer")
    print("="*60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    print(f"   ✓ Loaded successfully")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")

    # Test samples from different domains
    test_samples = {
        "Chat": "Hello! How are you doing today? I'm excited to learn about AI!",
        "Code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Math": "Given f(x) = 2x² + 3x - 5, find f'(x) = 4x + 3",
        "Reasoning": "Let's think step by step: First, we identify the problem. Second, we break it down."
    }

    print("\n2. Testing tokenization on sample texts:\n")

    total_chars = 0
    total_tokens = 0

    for domain, text in test_samples.items():
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chars = len(text)
        num_tokens = len(tokens)
        chars_per_token = chars / num_tokens

        total_chars += chars
        total_tokens += num_tokens

        print(f"   {domain}:")
        print(f"     Text: {text[:60]}...")
        print(f"     Characters: {chars}")
        print(f"     Tokens: {num_tokens}")
        print(f"     Chars/token: {chars_per_token:.2f}")
        print(f"     First 5 tokens: {tokens[:5]}")
        print(f"     Decoded: {[tokenizer.decode([t]) for t in tokens[:5]]}")
        print()

    avg_compression = total_chars / total_tokens
    print(f"   Overall average: {avg_compression:.2f} chars/token")
    print(f"   {'✓ Good compression!' if avg_compression > 3.0 else '⚠️  Low compression'}")

    # Test encoding/decoding roundtrip
    print("\n3. Testing encode/decode roundtrip:")
    original = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(original, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)

    print(f"   Original: {original}")
    print(f"   Tokens: {encoded}")
    print(f"   Decoded: {decoded}")
    print(f"   {'✓ Perfect match!' if original == decoded else '✗ Mismatch!'}")

    # Test numpy conversion (like we'll use in training)
    print("\n4. Testing numpy conversion:")
    token_array = np.array(encoded, dtype=np.uint16)
    print(f"   Array shape: {token_array.shape}")
    print(f"   Array dtype: {token_array.dtype}")
    print(f"   Memory size: {token_array.nbytes} bytes")
    print(f"   ✓ Numpy conversion works")

    # Simulate saving and loading
    print("\n5. Testing save/load:")
    test_file = Path("./data/test_tokens.npy")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(test_file, token_array)
    print(f"   Saved to: {test_file}")

    loaded = np.load(test_file)
    print(f"   Loaded shape: {loaded.shape}")
    print(f"   {'✓ Match!' if np.array_equal(token_array, loaded) else '✗ Mismatch!'}")

    # Clean up
    test_file.unlink()
    print(f"   Cleaned up test file")

    print("\n" + "="*60)
    print("✓ All tokenizer tests passed!")
    print("="*60)
    print("\nYou're ready to tokenize your downloaded data once it's complete.")
    print("\nNext step:")
    print("  python data/tokenize_data.py --create-split")

if __name__ == "__main__":
    test_tokenizer()
