#!/usr/bin/env python3
"""
Summarise a benchmark results directory as one comparison table.

Reads the *_{raw,chat}_*.json files written by run_benchmarks_direct.py and
groups them by (checkpoint, prompt_format), keeping the most recent run of each.
Result files predating the --chat-template work carry no "prompt_format" field
and are ignored, so old numbers can't silently mix into the comparison.

Usage:
    python3 eval/compare_results.py [results_dir]
"""

import json
import sys
from pathlib import Path

TASK_ORDER = ["hellaswag", "arc_easy", "arc_challenge", "winogrande",
              "piqa", "mmlu", "gsm8k", "humaneval"]

# Chance level, for reading a number as "learned something" vs "coin flip"
CHANCE = {"hellaswag": 25.0, "arc_easy": 25.0, "arc_challenge": 25.0,
          "winogrande": 50.0, "piqa": 50.0, "mmlu": 25.0,
          "gsm8k": 0.0, "humaneval": 0.0}


def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results")
    if not results_dir.is_dir():
        sys.exit(f"not a directory: {results_dir}")

    runs = {}
    skipped = 0
    for path in sorted(results_dir.glob("*.json")):
        if "_examples_" in path.name:
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "prompt_format" not in data or "tasks" not in data:
            skipped += 1
            continue
        key = (data["checkpoint"], data["prompt_format"])
        # sorted() walks timestamps in order, so the last write wins
        runs[key] = data

    if not runs:
        sys.exit(f"no comparable results in {results_dir}"
                 f"{f' ({skipped} pre-chat-template file(s) ignored)' if skipped else ''}")

    cols = sorted(runs, key=lambda k: (k[0], k[1] != "raw"))
    labels = [f"{Path(c).parent.name}/{f}" for c, f in cols]
    width = max(14, max(len(l) for l in labels) + 2)

    print(f"\n{'task':<16}" + "".join(f"{l:>{width}}" for l in labels) + f"{'chance':>9}")
    print("-" * (16 + width * len(cols) + 9))

    tasks = [t for t in TASK_ORDER if any(t in runs[k]["tasks"] for k in cols)]
    for task in tasks:
        row = f"{task:<16}"
        for key in cols:
            entry = runs[key]["tasks"].get(task)
            if entry is None:
                cell = "-"
            elif entry.get("note"):
                cell = "n/s"
            else:
                cell = f"{entry['accuracy']:.2f}"
            row += f"{cell:>{width}}"
        print(row + f"{CHANCE.get(task, 0.0):>8.0f}%")

    print("\n  n/s = not scored (humaneval never executes the generated code)")
    if skipped:
        print(f"  {skipped} result file(s) from before --chat-template ignored")

    # The comparison that answers "did SFT do anything on these benchmarks"
    base = next((k for k in cols if "sft" not in Path(k[0]).parent.name and k[1] == "raw"), None)
    sft = next((k for k in cols if "sft" in Path(k[0]).parent.name and k[1] == "chat"), None)
    if base and sft:
        print(f"\n  {'delta (sft/chat - base/raw)':<30}")
        for task in tasks:
            b, s = runs[base]["tasks"].get(task), runs[sft]["tasks"].get(task)
            if not b or not s or b.get("note") or s.get("note"):
                continue
            d = s["accuracy"] - b["accuracy"]
            print(f"    {task:<16}{d:+7.2f}")


if __name__ == "__main__":
    main()
