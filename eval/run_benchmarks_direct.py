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

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to use
        """
        self.device = device

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
        print(f"✓ Device: {device}, dtype: {self.dtype}\n")

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
            # Combine context and choice
            full_text = context + " " + choice
            full_texts.append(full_text)

            # Tokenize
            tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            context_tokens = self.tokenizer.encode(context, add_special_tokens=True)

            # Truncate if needed
            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length:]
                context_len = max(0, len(tokens) - (len(tokens) - len(context_tokens)))
            else:
                context_len = len(context_tokens)

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

    def evaluate_hellaswag(self, limit: int = None, save_examples: bool = False, examples_file: str = None):
        """Evaluate on HellaSwag benchmark."""
        from datasets import load_dataset

        print("Loading HellaSwag dataset...")
        dataset = load_dataset("Rowan/hellaswag", split="validation")

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        correct = 0
        total = 0
        examples_data = []

        print(f"Evaluating on {len(dataset)} examples...")

        for example in tqdm(dataset, desc="HellaSwag"):
            # Format context
            context = example["ctx"]

            # Get choices
            choices = example["endings"]

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
                tokens = self.tokenizer.encode(choice, add_special_tokens=True)
                if len(tokens) > self.max_length:
                    tokens = tokens[-self.max_length:]

                input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

                if self.use_autocast:
                    with torch.autocast(device_type=self.autocast_device, dtype=self.dtype):
                        logits, _ = self.model(input_ids)
                else:
                    logits, _ = self.model(input_ids)

                logits = logits[0]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Sum log probs for all tokens
                total_log_prob = 0.0
                for i in range(len(tokens) - 1):
                    token_id = tokens[i + 1]
                    total_log_prob += log_probs[i, token_id].item()

                # Normalize by length
                normalized = total_log_prob / len(tokens)
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

            # Format prompt for base model (not instruction-tuned)
            # Just give context and see if model generates the right number
            context = f"Q: {question}\nA: The answer is"

            # Generate completion
            input_ids = self.tokenizer.encode(context, add_special_tokens=True)
            input_ids = torch.tensor([input_ids[:self.model_config.block_size]]).to(self.device)

            # Generate a short completion
            with torch.no_grad():
                if self.use_autocast:
                    with torch.autocast(device_type=self.autocast_device, dtype=self.dtype):
                        output_ids = self.model.generate(
                            input_ids,
                            max_new_tokens=32,  # Short generation for answer
                            temperature=0.1,    # Low temperature for more deterministic
                        )
                else:
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=32,
                        temperature=0.1,
                    )

            # Decode the generated text
            generated_text = self.tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

            # Extract number from generated text
            pred_match = re.search(r"(-?\d+(?:,?\d+)*)", generated_text)

            is_correct = False
            pred_answer = ""

            if pred_match:
                pred_answer = pred_match.group(1).replace(",", "")
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

            # Generate code completion
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([input_ids[:self.model_config.block_size]]).to(self.device)

            with torch.no_grad():
                if self.use_autocast:
                    with torch.autocast(device_type=self.autocast_device, dtype=self.dtype):
                        output_ids = self.model.generate(
                            input_ids,
                            max_new_tokens=256,  # Enough for function body
                            temperature=0.2,
                        )
                else:
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=256,
                        temperature=0.2,
                    )

            # Decode the generated code
            generated_code = self.tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

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

    args = parser.parse_args()

    # Define task suites
    task_suites = {
        "quick": ["hellaswag"],
        "default": ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"],
        "full": ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa", "mmlu", "gsm8k", "humaneval"],
        "arc": ["arc_easy", "arc_challenge"],
        "instruction": ["mmlu", "gsm8k", "humaneval"],
    }

    # Get task list
    if args.tasks in task_suites:
        tasks = task_suites[args.tasks]
    else:
        tasks = args.tasks.split(",")

    print(f"\n{'='*80}")
    print(f"Picotalk Benchmark Evaluation")
    print(f"{'='*80}\n")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {tasks}")
    if args.limit:
        print(f"Limit: {args.limit} examples per task")
    print()

    # Load evaluator
    evaluator = BenchmarkEvaluator(args.checkpoint, device=args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations
    results = {}
    results["checkpoint"] = args.checkpoint
    results["timestamp"] = datetime.now().isoformat()
    results["tasks"] = {}

    # Prepare examples file path if saving
    ckpt_name = Path(args.checkpoint).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for task in tasks:
        print(f"\n{'='*80}")
        print(f"Running: {task.upper()}")
        print(f"{'='*80}\n")

        # Set up examples file if requested
        examples_file = None
        if args.save_examples:
            examples_file = output_dir / f"{ckpt_name}_{task}_examples_{timestamp}.json"

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
            print(f"Unknown task: {task}")
            continue

        results["tasks"][task] = task_results
        print(f"\n{task.upper()} Results: {task_results['accuracy']:.2f}% ({task_results['correct']}/{task_results['total']})")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for task, task_results in results["tasks"].items():
        print(f"{task:20s}: {task_results['accuracy']:6.2f}%  ({task_results['correct']:4d}/{task_results['total']:4d})")

    # Average
    avg_acc = np.mean([r["accuracy"] for r in results["tasks"].values()])
    print(f"\n{'Average':20s}: {avg_acc:6.2f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = Path(args.checkpoint).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{ckpt_name}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
