"""
Prepare the FULL smirki/UIGEN-T1.1-TAILWIND dataset for base model training.

Downloads all 805 samples from HuggingFace, splits into train/val/test,
converts to base format (prompt→code), and optionally pushes to HuggingFace.

This uses the full dataset (vs prepare_uigen_data.py which defaults to 100 train samples).

Usage:
    # Prepare and save locally only
    python finetuning/prepare_full_uigen_dataset.py

    # Prepare and push to HuggingFace (with confirmation)
    python finetuning/prepare_full_uigen_dataset.py --push --repo-name lilyzhng/uigen-ui-code-gen-full

    # Push without confirmation (for scripting)
    python finetuning/prepare_full_uigen_dataset.py --push --repo-name lilyzhng/uigen-ui-code-gen-full --yes
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

# Import the conversion logic from existing module
from finetuning.convert_uigen_to_base_format import convert_chatml_to_base_format

# Reproducible splits (same as prepare_uigen_data.py)
RANDOM_SEED = 42

# Output paths
OUTPUT_DIR = "data/base_format"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "uigen_train.jsonl")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "uigen_val.jsonl")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "uigen_test.jsonl")

# Split sizes (same ratio as prepare_uigen_data: 80 test, 10 val, rest train)
TEST_SIZE = 80
VAL_SIZE = 10


def convert_to_base_format(sample: dict) -> dict:
    """Convert smirki/UIGEN sample (question/answer) to base format."""
    # smirki dataset has: id, question, reasoning, answer
    # convert_chatml_to_base_format handles question/answer format
    return convert_chatml_to_base_format({
        "question": sample["question"],
        "answer": sample["answer"],
    })


def write_jsonl(data: list[dict], output_path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare full UIGEN-T1.1-TAILWIND dataset for base model training"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace after preparation",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="lilyzhng/uigen-ui-code-gen-full",
        help="HuggingFace dataset repo (default: lilyzhng/uigen-ui-code-gen-full)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation when pushing to HuggingFace",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for JSONL files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Override output paths if custom dir
    global TRAIN_OUTPUT, VAL_OUTPUT, TEST_OUTPUT
    TRAIN_OUTPUT = os.path.join(args.output_dir, "uigen_train.jsonl")
    VAL_OUTPUT = os.path.join(args.output_dir, "uigen_val.jsonl")
    TEST_OUTPUT = os.path.join(args.output_dir, "uigen_test.jsonl")

    print("=" * 80)
    print("Preparing FULL UIGEN-T1.1-TAILWIND dataset for base model training")
    print("=" * 80)
    print("Downloading smirki/UIGEN-T1.1-TAILWIND from HuggingFace...")

    dataset = load_dataset("smirki/UIGEN-T1.1-TAILWIND", split="train")
    total = len(dataset)
    print(f"Loaded {total} samples.")

    # Shuffle deterministically (same as prepare_uigen_data)
    indices = list(range(total))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Split: test first, then val, then train gets the rest
    test_indices = indices[:TEST_SIZE]
    val_indices = indices[TEST_SIZE : TEST_SIZE + VAL_SIZE]
    train_indices = indices[TEST_SIZE + VAL_SIZE :]

    train_count = len(train_indices)
    val_count = len(val_indices)
    test_count = len(test_indices)

    print(f"Split sizes — train: {train_count}, val: {val_count}, test: {test_count}")

    # Convert to base format and save
    train_data = [convert_to_base_format(dataset[i]) for i in train_indices]
    val_data = [convert_to_base_format(dataset[i]) for i in val_indices]
    test_data = [convert_to_base_format(dataset[i]) for i in test_indices]

    write_jsonl(train_data, TRAIN_OUTPUT)
    write_jsonl(val_data, VAL_OUTPUT)
    write_jsonl(test_data, TEST_OUTPUT)

    print(f"Saved {len(train_data)} train samples to {TRAIN_OUTPUT}")
    print(f"Saved {len(val_data)} val samples to {VAL_OUTPUT}")
    print(f"Saved {len(test_data)} test samples to {TEST_OUTPUT}")

    # Show example
    print("\nExample (first train sample, first 400 chars):")
    print("-" * 80)
    print(train_data[0]["text"][:400] + "...")
    print("-" * 80)

    # Push to HuggingFace if requested
    if args.push:
        from finetuning.push_to_huggingface import (
            create_dataset_dict,
            push_to_hub,
            create_dataset_card,
        )

        print("\nCreating dataset for HuggingFace...")
        dataset_dict = create_dataset_dict(args.output_dir)

        if not args.yes:
            response = input(f"Push dataset to {args.repo_name}? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("Aborted.")
                return

        push_to_hub(dataset_dict, args.repo_name, private=False)
        create_dataset_card(args.repo_name)
        print(f"\n✅ Dataset pushed to: https://huggingface.co/datasets/{args.repo_name}")

    print("\nDone! Use with Modal training:")
    print(f"  modal run unsloth/modal_coder_base.py --dataset-name {args.repo_name} --num-epochs 1")


if __name__ == "__main__":
    main()
