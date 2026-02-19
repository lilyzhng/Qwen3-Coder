"""
Prepare the UIGEN-T1.1-TAILWIND dataset for fine-tuning.

Downloads the dataset from HuggingFace, splits into train/val/test,
converts train+val to conversational JSONL format (ChatML), and saves the
test split with raw fields for evaluation.

Usage:
    python finetuning/prepare_uigen_data.py
    python finetuning/prepare_uigen_data.py --train-size 100
"""

import argparse
import json
import os
import random

from datasets import load_dataset

# Reproducible splits
RANDOM_SEED = 42

# Output paths (relative to project root)
OUTPUT_DIR = "data"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "uigen_train.jsonl")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "uigen_val.jsonl")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "uigen_test.jsonl")

# System prompt for the fine-tuning data
SYSTEM_PROMPT = (
    "You are an expert UI/UX developer. Generate clean, production-ready "
    "HTML and CSS code using Tailwind CSS."
)


def convert_to_chatml_format(sample: dict) -> dict:
    """Convert a UIGEN sample to ChatML conversational JSONL format.

    The assistant response is the code only (no reasoning).
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }


def convert_to_test_format(sample: dict) -> dict:
    """Keep raw fields for the test split so eval can compare against ground truth."""
    return {
        "id": sample["id"],
        "question": sample["question"],
        "reasoning": sample["reasoning"],
        "answer": sample["answer"],
    }


def write_jsonl(data: list[dict], output_path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare UIGEN dataset for fine-tuning")
    parser.add_argument(
        "--train-size", type=int, default=100,
        help="Number of training samples (default: 100)",
    )
    args = parser.parse_args()

    print("Downloading UIGEN-T1.1-TAILWIND dataset from HuggingFace...")
    dataset = load_dataset("smirki/UIGEN-T1.1-TAILWIND", split="train")
    total = len(dataset)
    print(f"Loaded {total} samples.")

    # Shuffle deterministically
    indices = list(range(total))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Reserve test and val first, then take train from the rest
    test_size = 80
    val_size = 10
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    remaining = indices[test_size + val_size :]
    train_indices = remaining[: args.train_size]

    print(f"Split sizes — train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")

    # Convert and save train set (code only, no reasoning)
    train_data = [convert_to_chatml_format(dataset[i]) for i in train_indices]
    write_jsonl(train_data, TRAIN_OUTPUT)
    print(f"Saved {len(train_data)} train samples to {TRAIN_OUTPUT}")

    # Convert and save val set
    val_data = [convert_to_chatml_format(dataset[i]) for i in val_indices]
    write_jsonl(val_data, VAL_OUTPUT)
    print(f"Saved {len(val_data)} val samples to {VAL_OUTPUT}")

    # Save test set with raw fields
    test_data = [convert_to_test_format(dataset[i]) for i in test_indices]
    write_jsonl(test_data, TEST_OUTPUT)
    print(f"Saved {len(test_data)} test samples to {TEST_OUTPUT}")

    print("\nDone! Data ready for training.")


if __name__ == "__main__":
    main()
