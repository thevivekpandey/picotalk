#!/usr/bin/env python3
"""
Direct benchmark evaluation for Picotalk using custom implementation.

This bypasses lm-eval's HuggingFace requirement and works directly with Picotalk checkpoints.

Usage:
    # Quick test (HellaSwag only, 100 examples)
    python eval/run_benchmarks_direct.py --checkpoint checkpoints/run_1b/step_45000.pt --limit 100

    # Full evaluation on standard suite
    python eval/run_benchmarks_direct.py --checkpoint checkpoints/run_1b/step_45000.pt --tasks default

    # Specific benchmarks
    python eval/run_benchmarks_direct.py --checkpoint checkpoints/run_1b/step_45000.pt --tasks hellaswag,arc_easy

    # SFT checkpoint, prompts wrapped in the [INST] format it was tuned on
    python eval/run_benchmarks_direct.py --checkpoint checkpoints/sft_1b/best.pt \
        --tasks full --chat-template

Prompt formats
--------------
Without --chat-template every prompt is raw text, which is what a pretrained
checkpoint saw. With it, prompts are wrapped as [INST] ... [/INST] and tokenized
segment-by-segment exactly as data/chat_format.py did during SFT.

Evaluating an SFT checkpoint WITHOUT the flag measures it out of distribution:
it was trained so that answers follow [/INST], and feeding it bare text hides
whatever the fine-tune installed. Compare base(raw) vs SFT(chat) -- and note the
results filename carries a `raw`/`chat` tag so the two never overwrite.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.model import Transformer
from transformers import AutoTokenizer


class BenchmarkEvaluator:
    """Direct benchmark evaluation for Picotalk."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", chat_template: bool = False):
        """
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to use
            chat_template: wrap every prompt in Mistral's [INST]...[/INST] format,
                matching what data/chat_format.py fed the model during SFT. Off by
                default: a pretrained checkpoint never saw [INST], so the base
                numbers stay directly comparable to earlier runs.
        """
        self.device = device
        self.chat_template = chat_template

        # Set dtype and autocast device based on hardware
        if device == "cuda":
            self.dtype = torch.bfloat16
            self.use_autocast = True
            self.autocast_device = "cuda"
        elif device == "mps":
            self.dtype = torch.float32  # Use float32 on MPS for numerical stability
            self.use_autocast = False  # MPS autocast is unstable
            self.autocast_device = None
        else:  # CPU
            self.dtype = torch.float32
            self.use_autocast = False
            self.autocast_device = None

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract config and create model
        from models.config import ModelConfig
        import inspect

        # Use model_config from checkpoint
        config_dict = checkpoint['model_config']

        # Filter to only valid ModelConfig parameters
        valid_params = set(inspect.signature(ModelConfig.__init__).parameters.keys())
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
        self.model_config = ModelConfig(**filtered_config)

        self.model = Transformer(self.model_config)

        # Load weights
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(device=device, dtype=self.dtype)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        self.max_length = self.model_config.block_size

        print(f"✓ Model loaded: {self.model.get_num_params()/1e6:.1f}M parameters")
        print(f"✓ Context length: {self.max_length}")
        print(f"✓ Device: {device}, dtype: {self.dtype}")
        print(f"✓ Prompt format: {'[INST] chat template' if chat_template else 'raw text'}\n")

    # ------------------------------------------------------------------
    # Prompt construction
    #
    # Both paths return (tokens, context_len) where tokens[context_len:] is the
    # span to score. The two differ in HOW they tokenize, and that difference is
    # deliberate:
    #
    #   raw text  - encode(context + " " + choice) as one string, which is how a
    #               pretrained model saw text. Kept byte-identical to the original
    #               implementation so base-model numbers remain comparable.
    #
    #   chat      - build the id list segment-by-segment exactly as
    #               data/chat_format.tokenize_conversation() did during SFT:
    #                 [BOS] + encode("[INST] {ctx} [/INST]") + encode(choice)
    #               Each segment is encoded with add_special_tokens=False, so
    #               SentencePiece's leading "▁" supplies the space after [/INST] --
    #               the same boundary the model was trained on. Building it
    #               segment-wise also makes context_len exact rather than inferred
    #               from a second encode() of the context alone.
    # ------------------------------------------------------------------

    def _encode_prompt(self, context: str) -> list:
        """Token ids for the prompt, up to (not including) the answer span."""
        if self.chat_template:
            return [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                f"[INST] {context.strip()} [/INST]", add_special_tokens=False
            )
        return self.tokenizer.encode(context, add_special_tokens=True)

    def _build_scored_ids(self, context: str, choice: str):
        """Return (tokens, context_len) for likelihood-scoring `choice`."""
        if self.chat_template:
            prompt_ids = self._encode_prompt(context)
            # The closing EOS is trained on but deliberately NOT scored here, so
            # the scored span is the answer text in both modes.
            choice_ids = self.tokenizer.encode(choice.strip(), add_special_tokens=False)
            tokens = prompt_ids + choice_ids
            context_len = len(prompt_ids)
        else:
            full_text = context + " " + choice
            tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            context_len = len(self.tokenizer.encode(context, add_special_tokens=True))

        # Left-truncate over-long inputs, keeping the answer span intact.
        if len(tokens) > self.max_length:
            overflow = len(tokens) - self.max_length
            tokens = tokens[overflow:]
            context_len = max(1, context_len - overflow)

        return tokens, context_len

    @torch.no_grad()
    def _generate_text(self, context: str, max_new_tokens: int, temperature: float) -> str:
        """Generate a continuation and decode it, stopping at the first EOS.

        Transformer.generate() has no stop-token support and always emits
        max_new_tokens, so an SFT model that correctly ends its turn with </s>
        would otherwise have whatever it rambled afterwards folded into the
        answer (skip_special_tokens=True hides the </s> but keeps the rambling).
        """
        prompt_ids = self._encode_prompt(context)[: self.max_length]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        if self.use_autocast:
            with torch.autocast(device_type=self.autocast_device, dtype=self.dtype):
                output_ids = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens, temperature=temperature
                )
        else:
            output_ids = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, temperature=temperature
            )

        new_ids = output_ids[0][len(prompt_ids):].tolist()
        if self.tokenizer.eos_token_id in new_ids:
            new_ids = new_ids[: new_ids.index(self.tokenizer.eos_token_id)]

        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    @torch.no_grad()
    def evaluate_multiple_choice(self, context: str, choices: list, return_details: bool = False):
        """
        Evaluate multiple choice by computing likelihood of each choice.

        Args:
            context: The question/context
            choices: List of possible completions
            return_details: If True, return detailed info about inputs and scores

        Returns:
            Index of most likely choice (or dict with details if return_details=True)
        """
        log_likelihoods = []
        full_texts = []

        for choice in choices:
            tokens, context_len = self._build_scored_ids(context, choice)
            full_texts.append(self.tokenizer.decode(tokens, skip_special_tokens=False))

            # Convert to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Get logits
            if self.use_autocast:
                with torch.autocast(device_type=self.autocast_device, dtype=self.dtype):
                    logits, _ = self.model(input_ids)
            else:
                logits, _ = self.model(input_ids)

            # Calculate log probability for choice tokens
            logits = logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Sum log probs for choice tokens (everything after context)
            choice_log_prob = 0.0
            for i in range(context_len, len(tokens)):
                token_id = tokens[i]
                if i - 1 >= 0:  # logits are shifted by 1
                    choice_log_prob += log_probs[i - 1, token_id].item()

            # Normalize by length (to avoid bias toward shorter choices)
            choice_length = len(tokens) - context_len
            if choice_length > 0:
                normalized_log_prob = choice_log_prob / choice_length
            else:
                normalized_log_prob = choice_log_prob

            log_likelihoods.append(normalized_log_prob)

        # Get prediction
        pred_idx = int(np.argmax(log_likelihoods))

        if return_details:
            return {
                "prediction": pred_idx,
                "full_texts": full_texts,  # Exact strings passed to model
                "log_likelihoods": log_likelihoods,  # Model scores for each choice
                "context": context,
                "choices": choices
            }
        else:
            return pred_idx

    @staticmethod
    def _hellaswag_clean(text: str) -> str:
        """lm-eval's HellaSwag preprocess(), reproduced verbatim.

        The wikiHow two-thirds of the set carries literal "[header]", "[title]",
        "[step]", "[substeps]" markup in both contexts and endings. No model has
        seen these in pretraining, so scoring them raw pushes that subset to
        chance. Every published HellaSwag number strips them this way.
        """
        import re
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    def evaluate_hellaswag(self, limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on HellaSwag benchmark."""
        from datasets import load_dataset

        print("Loading HellaSwag dataset...")
        dataset = load_dataset("Rowan/hellaswag", split="validation")

        # The validation file is not shuffled: 3,243 ActivityNet rows first,
        # then 6,799 wikiHow. Taking the head evaluates only the (easier,
        # markup-free) ActivityNet half. Sample evenly across the set instead.
        if limit and limit < len(dataset):
            step = len(dataset) / limit
            dataset = dataset.select([int(i * step) for i in range(limit)])

        correct = 0
        total = 0
        examples_data = []

        print(f"Evaluating on {len(dataset)} examples...")

        for example in tqdm(dataset, desc="HellaSwag"):
            # lm-eval's doc format: "{activity_label}: {ctx_a} {Ctx_b}", with
            # ctx_b capitalized (it is a sentence-initial pronoun like "he").
            ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
            context = self._hellaswag_clean(example["activity_label"] + ": " + ctx)

            # Endings carry the same markup and are cleaned the same way
            choices = [self._hellaswag_clean(e) for e in example["endings"]]

            # Get prediction (with details if saving examples)
            if save_examples:
                result = self.evaluate_multiple_choice(context, choices, return_details=True)
                pred_idx = result["prediction"]
            else:
                pred_idx = self.evaluate_multiple_choice(context, choices)

            # Check if correct
            label = int(example["label"])
            is_correct = pred_idx == label
            if is_correct:
                correct += 1
            total += 1

            # Save example details if requested
            if save_examples:
                examples_data.append({
                    "example_id": total,
                    "context": context,
                    "choices": choices,
                    "model_prediction": pred_idx,
                    "model_choice": choices[pred_idx],
                    "correct_label": label,
                    "correct_choice": choices[label],
                    "is_correct": is_correct,
                    "model_input_strings": result["full_texts"],  # Exact strings passed to model
                    "model_scores": result["log_likelihoods"]  # Log likelihood scores
                })

        accuracy = correct / total * 100

        # Save examples to file if requested
        if save_examples and examples_file:
            with open(examples_file, 'w') as f:
                import json
                json.dump(examples_data, f, indent=2)
            print(f"\nExamples saved to: {examples_file}")

        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_arc(self, subset: str = "easy", limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on ARC benchmark."""
        from datasets import load_dataset

        dataset_name = "ARC-Easy" if subset == "easy" else "ARC-Challenge"
        print(f"Loading {dataset_name} dataset...")
        dataset = load_dataset("allenai/ai2_arc", f"ARC-{subset.capitalize()}", split="test")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        correct = 0
        total = 0
        examples_data = []

        print(f"Evaluating on {len(dataset)} examples...")

        for example in tqdm(dataset, desc=dataset_name):
            # Format question
            question = example["question"]

            # Get choices
            choices = example["choices"]["text"]

            # Get prediction (with details if saving examples)
            if save_examples:
                result = self.evaluate_multiple_choice(question, choices, return_details=True)
                pred_idx = result["prediction"]
            else:
                pred_idx = self.evaluate_multiple_choice(question, choices)

            # Check if correct
            label_letter = example["answerKey"]
            # Convert letter to index (A=0, B=1, etc.)
            if label_letter.isdigit():
                label_idx = int(label_letter) - 1
            else:
                label_idx = ord(label_letter.upper()) - ord('A')

            is_correct = pred_idx == label_idx
            if is_correct:
                correct += 1
            total += 1

            # Save example details if requested
            if save_examples:
                examples_data.append({
                    "example_id": total,
                    "question": question,
                    "choices": choices,
                    "model_prediction": pred_idx,
                    "model_choice": choices[pred_idx],
                    "correct_label": label_idx,
                    "correct_answer": label_letter,
                    "correct_choice": choices[label_idx],
                    "is_correct": is_correct,
                    "model_input_strings": result["full_texts"],
                    "model_scores": result["log_likelihoods"]
                })

        accuracy = correct / total * 100

        # Save examples to file if requested
        if save_examples and examples_file:
            with open(examples_file, 'w') as f:
                import json
                json.dump(examples_data, f, indent=2)
            print(f"\nExamples saved to: {examples_file}")

        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_winogrande(self, limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on Winogrande benchmark."""
        from datasets import load_dataset

        print("Loading Winogrande dataset...")
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        correct = 0
        total = 0
        examples_data = []

        print(f"Evaluating on {len(dataset)} examples...")

        for example in tqdm(dataset, desc="Winogrande"):
            # Get sentence and options
            sentence = example["sentence"]
            option1 = example["option1"]
            option2 = example["option2"]

            # Create two versions with filled-in blanks
            choices = [
                sentence.replace("_", option1),
                sentence.replace("_", option2),
            ]

            # Evaluate which completion is more likely
            # For Winogrande, we evaluate the full sentence likelihood
            log_likelihoods = []
            for choice in choices:
                if self.chat_template:
                    # Present the blanked sentence as the instruction and score
                    # the filled-in sentence as the assistant's reply.
                    tokens, score_from = self._build_scored_ids(sentence, choice)
                    denom = len(tokens) - score_from
                else:
                    tokens = self.tokenizer.encode(choice, add_special_tokens=True)
                    if len(tokens) > self.max_length:
                        tokens = tokens[-self.max_length:]
                    # Score every token after BOS, normalised by the full length --
                    # the original arithmetic, preserved so base numbers don't move.
                    score_from = 1
                    denom = len(tokens)

                input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

                if self.use_autocast:
                    with torch.autocast(device_type=self.autocast_device, dtype=self.dtype):
                        logits, _ = self.model(input_ids)
                else:
                    logits, _ = self.model(input_ids)

                logits = logits[0]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Sum log probs over the scored span (logits are shifted by 1)
                total_log_prob = 0.0
                for i in range(score_from, len(tokens)):
                    total_log_prob += log_probs[i - 1, tokens[i]].item()

                # Normalize by length
                normalized = total_log_prob / max(1, denom)
                log_likelihoods.append(normalized)

            # Prediction
            pred_idx = int(np.argmax(log_likelihoods)) + 1  # 1-indexed

            # Check if correct
            label = int(example["answer"])
            is_correct = pred_idx == label
            if is_correct:
                correct += 1
            total += 1

            # Save example details if requested
            if save_examples:
                examples_data.append({
                    "example_id": total,
                    "sentence": sentence,
                    "option1": option1,
                    "option2": option2,
                    "choices": choices,
                    "model_prediction": pred_idx,
                    "model_choice": choices[pred_idx - 1],  # -1 because pred_idx is 1-indexed
                    "correct_label": label,
                    "correct_choice": choices[label - 1],  # -1 because label is 1-indexed
                    "is_correct": is_correct,
                    "model_input_strings": choices,
                    "model_scores": log_likelihoods
                })

        accuracy = correct / total * 100

        # Save examples to file if requested
        if save_examples and examples_file:
            with open(examples_file, 'w') as f:
                import json
                json.dump(examples_data, f, indent=2)
            print(f"\nExamples saved to: {examples_file}")

        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_piqa(self, limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on PIQA benchmark."""
        from datasets import load_dataset

        print("Loading PIQA dataset...")
        dataset = load_dataset("lighteval/piqa", split="validation")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        correct = 0
        total = 0
        examples_data = []

        print(f"Evaluating on {len(dataset)} examples...")

        for example in tqdm(dataset, desc="PIQA"):
            # Get goal (question) and solutions
            goal = example["goal"]
            sol1 = example["sol1"]
            sol2 = example["sol2"]

            # Format as multiple choice
            context = f"Question: {goal}\nAnswer:"
            choices = [sol1, sol2]

            # Get prediction (with details if saving examples)
            if save_examples:
                result = self.evaluate_multiple_choice(context, choices, return_details=True)
                pred_idx = result["prediction"]
            else:
                pred_idx = self.evaluate_multiple_choice(context, choices)

            # Check if correct
            label = int(example["label"])
            is_correct = pred_idx == label
            if is_correct:
                correct += 1
            total += 1

            # Save example details if requested
            if save_examples:
                examples_data.append({
                    "example_id": total,
                    "goal": goal,
                    "context": context,
                    "choices": choices,
                    "model_prediction": pred_idx,
                    "model_choice": choices[pred_idx],
                    "correct_label": label,
                    "correct_choice": choices[label],
                    "is_correct": is_correct,
                    "model_input_strings": result["full_texts"],
                    "model_scores": result["log_likelihoods"]
                })

        accuracy = correct / total * 100

        # Save examples to file if requested
        if save_examples and examples_file:
            with open(examples_file, 'w') as f:
                import json
                json.dump(examples_data, f, indent=2)
            print(f"\nExamples saved to: {examples_file}")

        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_mmlu(self, limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on MMLU benchmark (5-shot)."""
        from datasets import load_dataset

        print("Loading MMLU dataset...")
        # MMLU has multiple subjects, we'll test on a subset
        # Using the 'all' configuration which combines all subjects
        dataset = load_dataset("cais/mmlu", "all", split="test")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        correct = 0
        total = 0
        examples_data = []

        print(f"Evaluating on {len(dataset)} examples...")

        for example in tqdm(dataset, desc="MMLU"):
            # MMLU format: question with 4 choices (A, B, C, D)
            question = example["question"]
            choices = example["choices"]

            # Format as multiple choice
            context = f"Question: {question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

            # Get prediction (with details if saving examples)
            if save_examples:
                result = self.evaluate_multiple_choice(context, choices, return_details=True)
                pred_idx = result["prediction"]
            else:
                pred_idx = self.evaluate_multiple_choice(context, choices)

            # Check if correct
            label = example["answer"]  # Already 0-3 index
            is_correct = pred_idx == label
            if is_correct:
                correct += 1
            total += 1

            # Save example details if requested
            if save_examples:
                examples_data.append({
                    "example_id": total,
                    "question": question,
                    "context": context,
                    "choices": choices,
                    "model_prediction": pred_idx,
                    "model_choice": choices[pred_idx],
                    "correct_label": label,
                    "correct_choice": choices[label],
                    "is_correct": is_correct,
                    "model_input_strings": result["full_texts"],
                    "model_scores": result["log_likelihoods"]
                })

        accuracy = correct / total * 100

        # Save examples to file if requested
        if save_examples and examples_file:
            with open(examples_file, 'w') as f:
                import json
                json.dump(examples_data, f, indent=2)
            print(f"\nExamples saved to: {examples_file}")

        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_gsm8k(self, limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on GSM8K benchmark (math word problems)."""
        from datasets import load_dataset
        import re

        print("Loading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main", split="test")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        correct = 0
        total = 0
        examples_data = []

        print(f"Evaluating on {len(dataset)} examples...")

        for example in tqdm(dataset, desc="GSM8K"):
            question = example["question"]
            answer = example["answer"]

            # Extract numeric answer from the solution
            # GSM8K answers are in format "#### 42"
            match = re.search(r"####\s*(-?\d+(?:,?\d+)*)", answer)
            if not match:
                continue

            correct_answer = match.group(1).replace(",", "")

            # The two modes need genuinely different handling, because the two
            # models answer in different shapes:
            #
            #   raw  - a base model won't reason unprompted, so "A: The answer is"
            #          forces it to emit a number immediately. 32 tokens is plenty
            #          and the FIRST number is the answer.
            #   chat - an SFT model writes a worked solution. The "The answer is"
            #          cue is a base-model crutch and is dropped; it needs room to
            #          finish, and the LAST number is the answer (first-number
            #          extraction would grab an intermediate step and score ~0
            #          no matter how good the reasoning was).
            if self.chat_template:
                context = question
                generated_text = self._generate_text(context, max_new_tokens=256, temperature=0.1)
                numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", generated_text)
                pred_answer = numbers[-1].replace(",", "") if numbers else ""
            else:
                context = f"Q: {question}\nA: The answer is"
                generated_text = self._generate_text(context, max_new_tokens=32, temperature=0.1)
                match_pred = re.search(r"(-?\d+(?:,?\d+)*)", generated_text)
                pred_answer = match_pred.group(1).replace(",", "") if match_pred else ""

            # Compare as numbers so "72" and "72.0" agree
            is_correct = False
            if pred_answer:
                try:
                    is_correct = float(pred_answer) == float(correct_answer)
                except ValueError:
                    is_correct = pred_answer == correct_answer

            if is_correct:
                correct += 1
            total += 1

            # Save example details if requested
            if save_examples:
                examples_data.append({
                    "example_id": total,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_generation": generated_text,
                    "model_answer": pred_answer,
                    "is_correct": is_correct,
                })

        accuracy = correct / total * 100

        # Save examples to file if requested
        if save_examples and examples_file:
            with open(examples_file, 'w') as f:
                import json
                json.dump(examples_data, f, indent=2)
            print(f"\nExamples saved to: {examples_file}")

        return {"accuracy": accuracy, "correct": correct, "total": total}

    def evaluate_humaneval(self, limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on HumanEval benchmark (code generation)."""
        from datasets import load_dataset

        print("Loading HumanEval dataset...")
        dataset = load_dataset("openai/openai_humaneval", split="test")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Evaluating on {len(dataset)} examples...")
        print("Note: HumanEval requires code execution for proper evaluation.")
        print("This is a simplified version that generates code but doesn't execute it.")
        print("For full evaluation, use the official HumanEval evaluator.")

        examples_data = []

        for example in tqdm(dataset, desc="HumanEval"):
            prompt = example["prompt"]
            canonical_solution = example["canonical_solution"]
            test = example["test"]
            entry_point = example["entry_point"]

            # Generate code completion (max_new_tokens=256 fits a function body)
            generated_code = self._generate_text(prompt, max_new_tokens=256, temperature=0.2)

            # Save example details if requested
            if save_examples:
                examples_data.append({
                    "task_id": example["task_id"],
                    "prompt": prompt,
                    "canonical_solution": canonical_solution,
                    "model_generation": generated_code,
                    "entry_point": entry_point,
                })

        # For now, we can't properly evaluate without execution
        # Return 0% but save the generations for manual inspection
        accuracy = 0.0

        # Save examples to file if requested
        if save_examples and examples_file:
            with open(examples_file, 'w') as f:
                import json
                json.dump(examples_data, f, indent=2)
            print(f"\nExamples saved to: {examples_file}")
            print("To properly evaluate HumanEval, use: https://github.com/openai/human-eval")

        return {"accuracy": accuracy, "correct": 0, "total": len(dataset), "note": "Requires execution for proper evaluation"}


def main():
    parser = argparse.ArgumentParser(description="Direct benchmark evaluation for Picotalk")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="default",
        help="Tasks: 'default', 'quick', or comma-separated (hellaswag,arc_easy,arc_challenge,winogrande,piqa)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device to use (cuda/mps/cpu)",
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="Save input/output examples to file",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap prompts in Mistral [INST]...[/INST] format (use for SFT checkpoints)",
    )

    args = parser.parse_args()

    # Define task suites
    task_suites = {
        "quick": ["hellaswag"],
        "default": ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"],
        "full": ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa", "mmlu", "gsm8k", "humaneval"],
        "arc": ["arc_easy", "arc_challenge"],
        "instruction": ["mmlu", "gsm8k", "humaneval"],
    }

    known_tasks = task_suites["full"]

    # Expand the comma list, resolving suite names wherever they appear so
    # mixtures like "default,mmlu" work instead of being read as a task called
    # "default". Order is preserved and duplicates dropped.
    tasks = []
    unknown = []
    for piece in args.tasks.split(","):
        piece = piece.strip()
        if not piece:
            continue
        expanded = task_suites.get(piece, [piece])
        for task in expanded:
            if task not in known_tasks:
                unknown.append(task)
            elif task not in tasks:
                tasks.append(task)

    if unknown:
        parser.error(
            f"unknown task(s): {', '.join(unknown)}\n"
            f"  tasks:  {', '.join(known_tasks)}\n"
            f"  suites: {', '.join(task_suites)}"
        )
    if not tasks:
        parser.error("no tasks selected")

    print(f"\n{'='*80}")
    print(f"Picotalk Benchmark Evaluation")
    print(f"{'='*80}\n")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {tasks}")
    print(f"Prompt format: {'[INST] chat template' if args.chat_template else 'raw text'}")
    if args.limit:
        print(f"Limit: {args.limit} examples per task")
    print()

    # Load evaluator
    evaluator = BenchmarkEvaluator(
        args.checkpoint, device=args.device, chat_template=args.chat_template
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations
    results = {}
    results["checkpoint"] = args.checkpoint
    results["timestamp"] = datetime.now().isoformat()
    results["prompt_format"] = "chat" if args.chat_template else "raw"
    results["tasks"] = {}

    # Prepare examples file path if saving. The prompt format goes in the name so
    # a chat run never silently overwrites the raw run it is being compared with.
    # Include the checkpoint's directory: run_1b/best.pt and sft_1b/best.pt
    # both have the stem "best", so the stem alone cannot tell them apart.
    _ckpt = Path(args.checkpoint)
    ckpt_name = f"{_ckpt.parent.name}_{_ckpt.stem}" if _ckpt.parent.name else _ckpt.stem
    fmt_tag = "chat" if args.chat_template else "raw"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for task in tasks:
        print(f"\n{'='*80}")
        print(f"Running: {task.upper()}")
        print(f"{'='*80}\n")

        # Set up examples file if requested
        examples_file = None
        if args.save_examples:
            examples_file = output_dir / f"{ckpt_name}_{fmt_tag}_{task}_examples_{timestamp}.json"

        if task == "hellaswag":
            task_results = evaluator.evaluate_hellaswag(
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        elif task == "arc_easy":
            task_results = evaluator.evaluate_arc(
                subset="easy",
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        elif task == "arc_challenge":
            task_results = evaluator.evaluate_arc(
                subset="challenge",
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        elif task == "winogrande":
            task_results = evaluator.evaluate_winogrande(
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        elif task == "piqa":
            task_results = evaluator.evaluate_piqa(
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        elif task == "mmlu":
            task_results = evaluator.evaluate_mmlu(
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        elif task == "gsm8k":
            task_results = evaluator.evaluate_gsm8k(
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        elif task == "humaneval":
            task_results = evaluator.evaluate_humaneval(
                limit=args.limit,
                save_examples=args.save_examples,
                examples_file=str(examples_file) if examples_file else None
            )
        else:
            raise AssertionError(f"task {task!r} passed validation but has no handler")

        results["tasks"][task] = task_results
        print(f"\n{task.upper()} Results: {task_results['accuracy']:.2f}% ({task_results['correct']}/{task_results['total']})")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for task, task_results in results["tasks"].items():
        note = "  (not scored)" if task_results.get("note") else ""
        print(f"{task:20s}: {task_results['accuracy']:6.2f}%  "
              f"({task_results['correct']:4d}/{task_results['total']:4d}){note}")

    # Average over MEASURED tasks only. humaneval reports a hardcoded 0.0 because
    # it never executes the generated code, and folding that into the mean drags
    # the headline number down by a task nobody actually scored.
    measured = [r["accuracy"] for r in results["tasks"].values() if not r.get("note")]
    if measured:
        print(f"\n{'Average':20s}: {np.mean(measured):6.2f}%  "
              f"(over {len(measured)} scored task(s))")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{ckpt_name}_{fmt_tag}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
