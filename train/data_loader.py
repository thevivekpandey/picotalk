"""
Data loader for training Picotalk.

Loads tokenized .bin files using memory-mapped arrays for efficiency.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple


class TokenDataset:
    """Memory-mapped token dataset."""

    def __init__(self, bin_files: List[Path]):
        """
        Args:
            bin_files: list of .bin files to load
        """
        self.data_files = []
        self.cumulative_lengths = [0]
        total_tokens = 0

        for path in bin_files:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

            # Memory-map the file
            tokens = np.memmap(path, dtype=np.uint16, mode='r')
            self.data_files.append(tokens)

            total_tokens += len(tokens)
            self.cumulative_lengths.append(total_tokens)

        self.total_tokens = total_tokens
        self.file_lengths = np.array([len(f) for f in self.data_files], dtype=np.int64)
        print(f"Loaded {len(bin_files)} files, {total_tokens:,} total tokens")

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of data.

        Args:
            batch_size: number of sequences
            block_size: sequence length
            device: device to place tensors on
            generator: optional RNG; pass one seeded per (rank, step) to make
                the data stream reproducible and resumable

        Returns:
            x: (batch_size, block_size) input tokens
            y: (batch_size, block_size) target tokens
        """
        # A window needs block_size + 1 tokens (inputs plus the targets shifted
        # by one), so per file the valid starts are 0 .. len - block_size - 1.
        # Sampling within a file (rather than over the concatenated stream)
        # keeps windows from straddling a file boundary and yielding short
        # sequences, which would fail the torch.stack below.
        valid_starts = np.maximum(self.file_lengths - block_size, 0)
        total_valid = int(valid_starts.sum())

        if total_valid == 0:
            raise ValueError(
                f"No file is long enough for block_size={block_size} "
                f"(longest is {int(self.file_lengths.max()):,} tokens)"
            )

        # Weight each file by its valid-start count, so every window in the
        # dataset remains equally likely.
        cum_valid = np.cumsum(valid_starts)
        offsets = torch.randint(total_valid, (batch_size,), generator=generator).numpy()
        file_indices = np.searchsorted(cum_valid, offsets, side='right')
        local_indices = offsets - (cum_valid[file_indices] - valid_starts[file_indices])

        x_list = []
        y_list = []

        for file_idx, local_idx in zip(file_indices, local_indices):
            # Extract sequence
            data_file = self.data_files[file_idx]
            x_seq = torch.from_numpy(data_file[local_idx:local_idx + block_size].astype(np.int64))
            y_seq = torch.from_numpy(data_file[local_idx + 1:local_idx + block_size + 1].astype(np.int64))

            x_list.append(x_seq)
            y_list.append(y_seq)

        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)

        return x, y


def create_dataloaders(data_dir: Path, batch_size: int, block_size: int, device: str = 'cuda'):
    """
    Create train and val dataloaders.

    Args:
        data_dir: directory containing *_train.bin and *_val.bin files
        batch_size: batch size
        block_size: sequence length
        device: device to use

    Returns:
        train_dataset, val_dataset
    """
    train_files = sorted(data_dir.glob("*_train.bin"))
    val_files = sorted(data_dir.glob("*_val.bin"))

    if not train_files:
        raise ValueError(f"No *_train.bin files found in {data_dir}")

    print(f"\nTrain files: {[f.name for f in train_files]}")
    print(f"Val files: {[f.name for f in val_files]}")

    train_dataset = TokenDataset(train_files)
    val_dataset = TokenDataset(val_files) if val_files else None

    return train_dataset, val_dataset
