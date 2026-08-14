"""
Data loader for training Picotalk.

Loads tokenized .bin files using memory-mapped arrays for efficiency.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple


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
        print(f"Loaded {len(bin_files)} files, {total_tokens:,} total tokens")

    def get_batch(self, batch_size: int, block_size: int, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of data.

        Args:
            batch_size: number of sequences
            block_size: sequence length
            device: device to place tensors on

        Returns:
            x: (batch_size, block_size) input tokens
            y: (batch_size, block_size) target tokens
        """
        # Sample random starting positions
        ix = torch.randint(self.total_tokens - block_size, (batch_size,))

        x_list = []
        y_list = []

        for i in ix:
            # Find which file this index belongs to
            file_idx = 0
            for j, cum_len in enumerate(self.cumulative_lengths[1:]):
                if i < cum_len:
                    file_idx = j
                    break

            # Get local index within that file
            local_idx = i - self.cumulative_lengths[file_idx]

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
