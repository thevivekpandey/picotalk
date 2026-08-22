"""
SFT data loader for Picotalk.

Like train/data_loader.py, but each token stream has a parallel uint8 loss
mask (produced by data/prepare_sft_data.py). Targets where mask == 0 (BOS,
user/system prompt tokens) are set to -1, which the model's cross_entropy
ignores (ignore_index=-1). Only assistant tokens and their closing EOS
contribute to the loss.

Files expected in the data directory:
    sft_train_tokens.bin  uint16
    sft_train_mask.bin    uint8   (same length as tokens)
    sft_val_tokens.bin
    sft_val_mask.bin
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

IGNORE_INDEX = -1


class SFTDataset:
    """Memory-mapped token + loss-mask dataset."""

    def __init__(self, tokens_file: Path, mask_file: Path):
        if not tokens_file.exists():
            raise FileNotFoundError(f"Tokens file not found: {tokens_file}")
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_file}")

        self.tokens = np.memmap(tokens_file, dtype=np.uint16, mode='r')
        self.mask = np.memmap(mask_file, dtype=np.uint8, mode='r')

        if len(self.tokens) != len(self.mask):
            raise ValueError(
                f"Length mismatch: {tokens_file.name} has {len(self.tokens):,} "
                f"tokens but {mask_file.name} has {len(self.mask):,} entries"
            )

        self.total_tokens = len(self.tokens)
        trainable = int(np.asarray(self.mask).sum())
        print(f"Loaded {tokens_file.name}: {self.total_tokens:,} tokens "
              f"({trainable:,} trainable, {trainable / self.total_tokens * 100:.1f}%)")

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random batch with loss-masked targets.

        Returns:
            x: (batch_size, block_size) input tokens
            y: (batch_size, block_size) targets; -1 where the loss is masked
        """
        n_valid = self.total_tokens - block_size
        if n_valid <= 0:
            raise ValueError(
                f"Dataset too small for block_size={block_size} "
                f"({self.total_tokens:,} tokens)"
            )

        starts = torch.randint(n_valid, (batch_size,), generator=generator).numpy()

        x_list = []
        y_list = []
        for i in starts:
            # A window with zero trainable targets contributes nothing and, if
            # a whole batch were masked, cross_entropy would return nan. Rare
            # with 2048-token windows, but tiny smoke-test splits can hit it.
            for _ in range(10):
                m_seq = torch.from_numpy(self.mask[i + 1:i + block_size + 1].astype(np.int64))
                if m_seq.sum() > 0:
                    break
                i = int(torch.randint(n_valid, (1,), generator=generator))

            x_seq = torch.from_numpy(self.tokens[i:i + block_size].astype(np.int64))
            y_seq = torch.from_numpy(self.tokens[i + 1:i + block_size + 1].astype(np.int64))
            y_seq[m_seq == 0] = IGNORE_INDEX
            x_list.append(x_seq)
            y_list.append(y_seq)

        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        return x, y


def create_sft_dataloaders(data_dir: Path):
    """Create train and val SFTDatasets from a prepare_sft_data.py output dir."""
    train_dataset = SFTDataset(
        data_dir / "sft_train_tokens.bin",
        data_dir / "sft_train_mask.bin",
    )

    val_tokens = data_dir / "sft_val_tokens.bin"
    val_mask = data_dir / "sft_val_mask.bin"
    val_dataset = SFTDataset(val_tokens, val_mask) if val_tokens.exists() else None

    return train_dataset, val_dataset
