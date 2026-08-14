#!/usr/bin/env python3
"""
Quick test to verify Picotalk setup is working correctly.

Tests:
1. Model instantiation
2. Data loading from tokenized files
3. Forward pass (single batch)
4. Training step (sanity check)
"""

import torch
import numpy as np
from pathlib import Path

from models.config import get_config
from models.model import build_model
from train.data_loader import create_dataloaders


def test_model():
    """Test model instantiation and forward pass."""
    print("="*60)
    print("Test 1: Model Instantiation")
    print("="*60)

    # Test small model first
    config = get_config("test")
    model = build_model(config)

    print(f"✓ Model created: {model.get_num_params()/1e6:.2f}M params")

    # Test forward pass
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    return True


def test_data_loader():
    """Test data loading from tokenized files."""
    print("\n" + "="*60)
    print("Test 2: Data Loading")
    print("="*60)

    data_dir = Path("./data/tokenized")

    # Check if tokenized data exists
    bin_files = list(data_dir.glob("*_train.bin"))

    if not bin_files:
        print("⚠️  No tokenized data found in ./data/tokenized")
        print("   Run: python3 data/tokenize_data.py")
        return False

    print(f"Found {len(bin_files)} tokenized datasets:")
    for f in bin_files:
        # Check file size
        size_mb = f.stat().st_size / 1024**2
        tokens = np.memmap(f, dtype=np.uint16, mode='r')
        print(f"  {f.stem}: {len(tokens):,} tokens ({size_mb:.1f} MB)")

    # Test data loader
    try:
        train_dataset, val_dataset = create_dataloaders(
            data_dir,
            batch_size=4,
            block_size=256,
            device='cpu'
        )

        print(f"\n✓ DataLoader created")
        print(f"  Total train tokens: {train_dataset.total_tokens:,}")
        if val_dataset:
            print(f"  Total val tokens: {val_dataset.total_tokens:,}")

        # Get a batch
        x, y = train_dataset.get_batch(4, 256, 'cpu')
        print(f"\n✓ Batch loading successful")
        print(f"  Batch shape: {x.shape}")
        print(f"  Sample tokens: {x[0, :10].tolist()}")

        return True

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("Test 3: Training Step")
    print("="*60)

    data_dir = Path("./data/tokenized")
    if not list(data_dir.glob("*_train.bin")):
        print("⚠️  Skipping (no tokenized data)")
        return False

    # Small model for fast test
    config = get_config("test")
    model = build_model(config)

    # Data loader
    train_dataset, _ = create_dataloaders(
        data_dir,
        batch_size=2,
        block_size=256,
        device='cpu'
    )

    # Get batch
    x, y = train_dataset.get_batch(2, 256, 'cpu')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training step
    model.train()
    optimizer.zero_grad()

    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {model.layers[0].attention.wq.weight.grad is not None}")

    return True


def test_generation():
    """Test text generation."""
    print("\n" + "="*60)
    print("Test 4: Text Generation")
    print("="*60)

    from transformers import AutoTokenizer

    # Small model
    config = get_config("test")
    model = build_model(config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Test prompt
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate (untrained model, will be random)
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"✓ Generation successful")
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {generated_text[:100]}...")

    return True


def test_checkpoint():
    """Test checkpoint saving and loading."""
    print("\n" + "="*60)
    print("Test 5: Checkpoint Save/Load")
    print("="*60)

    import tempfile

    # Create model
    config = get_config("test")
    model = build_model(config)

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name

    checkpoint = {
        'step': 100,
        'model_state_dict': model.state_dict(),
        'config': vars(config),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Load checkpoint
    loaded = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(loaded['model_state_dict'])

    print(f"✓ Checkpoint loaded")
    print(f"  Step: {loaded['step']}")
    print(f"  Config: {loaded['config']['n_layer']}L, {loaded['config']['n_embd']}d")

    # Clean up
    Path(checkpoint_path).unlink()

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Picotalk Setup Test")
    print("="*60)

    tests = [
        ("Model Instantiation", test_model),
        ("Data Loading", test_data_loader),
        ("Training Step", test_training_step),
        ("Text Generation", test_generation),
        ("Checkpoint Save/Load", test_checkpoint),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n🎉 All tests passed! Picotalk setup is ready.")
        print("\nNext steps:")
        print("1. Download datasets: python3 data/download_datasets.py --datasets all")
        print("2. Tokenize data: python3 data/tokenize_data.py")
        print("3. Run scaling experiments: python3 train/scaling_experiments.py --list")
        print("4. Train model: python3 train/train.py --model 1b")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
