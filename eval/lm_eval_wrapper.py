"""
LM Evaluation Harness wrapper for Picotalk models.

This adapter allows Picotalk models to be evaluated using EleutherAI's lm-evaluation-harness.
"""

import torch
from typing import Optional, List, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.model import Transformer
from models.config import get_config
from transformers import AutoTokenizer


class PicotalkLM:
    """Wrapper for Picotalk model to work with lm-eval."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            dtype: Data type for model weights
        """
        self.device = device
        self.dtype = dtype

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract config and create model
        model_config = checkpoint['config']
        self.model = Transformer(model_config)

        # Load weights
        state_dict = checkpoint['model']
        # Remove 'module.' prefix if present (from DDP)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        # Move to device and set dtype
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        # Model properties
        self.vocab_size = model_config.vocab_size
        self.max_length = model_config.block_size

        print(f"Model loaded: {self.model.get_num_params()/1e6:.1f}M parameters")
        print(f"Context length: {self.max_length}")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Device: {device}, dtype: {dtype}")

    @torch.no_grad()
    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for context-continuation pairs.

        Args:
            requests: List of (context, continuation) tuples

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        for context, continuation in requests:
            # Tokenize
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)

            # Combine
            full_tokens = context_tokens + continuation_tokens

            # Truncate if too long
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[-self.max_length:]
                context_len = max(0, len(full_tokens) - len(continuation_tokens))
            else:
                context_len = len(context_tokens)

            # Convert to tensor
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=self.device)

            # Get logits
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                logits, _ = self.model(input_ids)

            # Calculate log probabilities for continuation tokens
            logits = logits[0]  # Remove batch dimension
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Sum log probabilities for continuation tokens
            total_log_prob = 0.0
            is_greedy = True

            for i, token_id in enumerate(continuation_tokens):
                position = context_len + i - 1  # -1 because logits are shifted
                if position >= 0 and position < len(log_probs):
                    token_log_prob = log_probs[position, token_id].item()
                    total_log_prob += token_log_prob

                    # Check if this token is greedy choice
                    greedy_token = log_probs[position].argmax().item()
                    if greedy_token != token_id:
                        is_greedy = False

            results.append((total_log_prob, is_greedy))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: List[str]) -> List[float]:
        """
        Compute log-likelihood for full sequences (rolling).

        Args:
            requests: List of text sequences

        Returns:
            List of log-likelihoods
        """
        results = []

        for text in requests:
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=True)

            # Truncate if too long
            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length:]

            # Convert to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Get logits
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                logits, _ = self.model(input_ids)

            # Calculate log probabilities
            logits = logits[0]  # Remove batch dimension
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Sum log probabilities for all tokens
            total_log_prob = 0.0
            for i in range(len(tokens) - 1):
                token_id = tokens[i + 1]
                token_log_prob = log_probs[i, token_id].item()
                total_log_prob += token_log_prob

            results.append(total_log_prob)

        return results

    @torch.no_grad()
    def generate_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        """
        Generate continuations until stopping criteria.

        Args:
            requests: List of (context, generation_kwargs) tuples

        Returns:
            List of generated continuations
        """
        results = []

        for context, gen_kwargs in requests:
            # Tokenize context
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)

            # Truncate if too long
            if len(context_tokens) > self.max_length - 100:  # Leave room for generation
                context_tokens = context_tokens[-(self.max_length - 100):]

            # Convert to tensor
            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=self.device)

            # Generation parameters
            max_gen_len = gen_kwargs.get('max_gen_toks', 256)
            until = gen_kwargs.get('until', [self.tokenizer.eos_token])
            temperature = gen_kwargs.get('temperature', 0.0)  # Greedy by default

            # Generate
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen_len,
                    temperature=max(temperature, 1e-7),  # Avoid 0
                    top_k=1 if temperature == 0.0 else 50,
                )

            # Decode only the generated part
            generated_tokens = output_ids[0, len(context_tokens):].tolist()
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Truncate at stopping sequences
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]

            results.append(generated_text)

        return results

    def eot_token_id(self) -> int:
        """Return end-of-text token ID."""
        return self.tokenizer.eos_token_id

    def max_seq_length(self) -> int:
        """Return maximum sequence length."""
        return self.max_length

    def vocab_size_property(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


def load_picotalk_model(checkpoint_path: str, device: str = "cuda") -> PicotalkLM:
    """
    Load Picotalk model for evaluation.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on

    Returns:
        PicotalkLM wrapper instance
    """
    return PicotalkLM(checkpoint_path=checkpoint_path, device=device)
