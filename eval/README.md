# Picotalk Benchmark Evaluation

Direct benchmark evaluation for Picotalk models without requiring HuggingFace format conversion.

## Features

- **Direct checkpoint loading**: Works with raw PyTorch checkpoints
- **Cross-platform support**: Automatic device detection (CUDA, MPS, CPU)
- **Detailed logging**: Save exact model inputs and outputs for analysis
- **5 Standard benchmarks**: HellaSwag, ARC-Easy, ARC-Challenge, Winogrande, PIQA

## Usage

### Quick Test (HellaSwag only)
```bash
python3 eval/run_benchmarks_direct.py \
  --checkpoint checkpoints/run_1b/step_45000.pt \
  --tasks quick \
  --limit 10 \
  --device mps
```

### All Benchmarks (Default Suite)
```bash
python3 eval/run_benchmarks_direct.py \
  --checkpoint checkpoints/run_1b/step_45000.pt \
  --tasks default \
  --device cuda
```

### Specific Benchmarks
```bash
python3 eval/run_benchmarks_direct.py \
  --checkpoint checkpoints/run_1b/step_45000.pt \
  --tasks hellaswag,arc_easy,winogrande \
  --device cuda
```

### Save Detailed Examples
```bash
python3 eval/run_benchmarks_direct.py \
  --checkpoint checkpoints/run_1b/step_45000.pt \
  --tasks default \
  --device cuda \
  --save-examples
```

## Command-line Options

- `--checkpoint`: Path to PyTorch checkpoint file (required)
- `--tasks`: Task suite or comma-separated task list
  - Task suites: `quick`, `default`, `arc`
  - Individual tasks: `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, `piqa`
- `--device`: Device to use (`cuda`, `mps`, `cpu`), auto-detected if not specified
- `--limit`: Limit number of examples per task (for testing)
- `--output-dir`: Directory for results (default: `eval/results`)
- `--save-examples`: Save detailed input/output examples to JSON files

## Benchmarks

### HellaSwag
- **Task**: Commonsense reasoning about everyday events
- **Format**: Context + 4 possible continuations
- **Dataset**: Rowan/hellaswag
- **Split**: validation (10,042 examples)

### ARC (AI2 Reasoning Challenge)
- **Task**: Grade-school science questions
- **Format**: Question + 3-5 multiple choice answers
- **Subsets**:
  - ARC-Easy: allenai/ai2_arc/ARC-Easy (2,376 validation examples)
  - ARC-Challenge: allenai/ai2_arc/ARC-Challenge (1,172 test examples)

### Winogrande
- **Task**: Pronoun resolution and commonsense reasoning
- **Format**: Sentence with blank + 2 options
- **Dataset**: allenai/winogrande
- **Split**: validation (1,267 examples)

### PIQA (Physical Interaction QA)
- **Task**: Physical commonsense reasoning
- **Format**: Goal + 2 possible solutions
- **Dataset**: lighteval/piqa
- **Split**: validation (1,838 examples)

## Output Files

Results are saved to `eval/results/` with timestamps:

### Summary Results
`step_45000_20260820_114046.json`:
```json
{
  "checkpoint": "checkpoints/run_1b/step_45000.pt",
  "timestamp": "2026-08-20T11:40:46.123456",
  "tasks": {
    "hellaswag": {"accuracy": 50.0, "correct": 1, "total": 2},
    "arc_easy": {"accuracy": 50.0, "correct": 1, "total": 2},
    "piqa": {"accuracy": 100.0, "correct": 2, "total": 2}
  }
}
```

### Detailed Examples (with --save-examples)
`step_45000_hellaswag_examples_20260820_114023.json`:
```json
[
  {
    "example_id": 1,
    "context": "A man is sitting on a roof. he",
    "choices": [
      "is using wrap to wrap a pair of skis.",
      "starts pulling up roofing on a roof."
    ],
    "model_prediction": 0,
    "model_choice": "is using wrap to wrap a pair of skis.",
    "correct_label": 1,
    "correct_choice": "starts pulling up roofing on a roof.",
    "is_correct": false,
    "model_input_strings": [
      "A man is sitting on a roof. he is using wrap to wrap a pair of skis.",
      "A man is sitting on a roof. he starts pulling up roofing on a roof."
    ],
    "model_scores": [-4.99, -3.89]
  }
]
```

## Device-Specific Notes

### CUDA
- Uses bfloat16 for faster inference
- Automatic mixed precision enabled

### MPS (Apple Silicon)
- Uses float32 for numerical stability
- Float16 causes identical scores across choices

### CPU
- Uses float32
- Slower but works on any machine

## Scoring Method

For multiple-choice tasks:
1. Concatenate context + each choice
2. Tokenize and feed to model
3. Calculate log-probability for tokens in the choice part only
4. Normalize by choice length: `score = sum(log_probs) / num_tokens`
5. Select choice with highest normalized score

This ensures:
- Only the answer is scored, not the question
- Shorter choices don't have unfair advantage
- Works for next-token prediction models without instruction-following
