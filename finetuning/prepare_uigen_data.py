"""
Prepare UIGEN datasets for fine-tuning: train/val/test split and ChatML JSONL.

Supports:
  - smirki/UIGEN-T1.1-TAILWIND (columns: question, answer, reasoning, id) — 805 samples
  - Tesslate/UIGEN-T2 (columns: prompt, reasoning, response)

Usage:
    # T1.1 full dataset: 645 train / 80 val / 80 test (uses all 805 samples)
    python finetuning/prepare_uigen_data.py

    # T1.1 full dataset + upload to HuggingFace as lilyzhng/UIGEN-T1.1-split
    python finetuning/prepare_uigen_data.py --upload

    # Custom HF repo name
    python finetuning/prepare_uigen_data.py --upload --hf-repo myuser/my-dataset

    # UIGEN-T2 subset: 1K examples → 800 train, 100 val, 100 test
    python finetuning/prepare_uigen_data.py --dataset Tesslate/UIGEN-T2 --max-samples 1000

    # UIGEN-T2 with custom split (e.g. 700 train, 150 val, 150 test)
    python finetuning/prepare_uigen_data.py --dataset Tesslate/UIGEN-T2 --max-samples 1000 --val-size 150 --test-size 150
"""

import argparse
import json
import os
import random

from datasets import load_dataset

# Reproducible splits
RANDOM_SEED = 42

# Dataset configs: HF id -> (split name, user_col, assistant_col, reasoning_col, id_col or None)
DATASET_CONFIG = {
    "smirki/UIGEN-T1.1-TAILWIND": {
        "split": "train",
        "user_col": "question",
        "assistant_col": "answer",
        "reasoning_col": "reasoning",
        "id_col": "id",
    },
    "Tesslate/UIGEN-T2": {
        "split": "train",
        "user_col": "prompt",
        "assistant_col": "response",
        "reasoning_col": "reasoning",
        "id_col": None,  # no id column; we use index
    },
}

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


def normalize_sample(sample: dict, index: int, config: dict) -> dict:
    """Normalize a row to common keys: question, answer, reasoning, id."""
    user_col = config["user_col"]
    assistant_col = config["assistant_col"]
    reasoning_col = config["reasoning_col"]
    id_col = config["id_col"]
    return {
        "question": sample[user_col],
        "answer": sample[assistant_col],
        "reasoning": sample.get(reasoning_col) or "",
        "id": sample.get(id_col) if id_col else index,
    }


def convert_to_chatml_format(normalized: dict) -> dict:
    """Convert a normalized sample to ChatML conversational JSONL format.

    The assistant response is the code only (no reasoning).
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalized["question"]},
            {"role": "assistant", "content": normalized["answer"]},
        ]
    }


def convert_to_test_format(normalized: dict) -> dict:
    """Keep raw fields for the test split so eval can compare against ground truth."""
    return {
        "id": normalized["id"],
        "question": normalized["question"],
        "reasoning": normalized["reasoning"],
        "answer": normalized["answer"],
    }


def write_jsonl(data: list[dict], output_path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare UIGEN dataset for fine-tuning")
    parser.add_argument(
        "--dataset",
        type=str,
        default="smirki/UIGEN-T1.1-TAILWIND",
        choices=list(DATASET_CONFIG.keys()),
        help="HuggingFace dataset ID (default: smirki/UIGEN-T1.1-TAILWIND)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max examples to use (e.g. 1000 for UIGEN-T2 subset). If set, we shuffle then take this many before splitting.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training samples. If not set: for T1.1 uses 100; for 1K subset uses 800.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Validation set size (default: 10 for T1.1, 100 for 1K subset)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Test set size (default: 80 for T1.1, 100 for 1K subset)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the split dataset to HuggingFace Hub as a DatasetDict with train/val/test splits",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="lilyzhng/UIGEN-T1.1-split",
        help="HuggingFace repo ID to upload to (default: lilyzhng/UIGEN-T1.1-split)",
    )
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset]

    # Default split sizes
    if args.max_samples:
        # Subset mode: 80/10/10 split
        default_train = int(args.max_samples * 0.8)
        default_val = int(args.max_samples * 0.1)
        default_test = args.max_samples - default_train - default_val
    else:
        # Full T1.1 dataset (805 samples): 645 train / 80 val / 80 test
        default_train = 645
        default_val = 80
        default_test = 80

    train_size = args.train_size if args.train_size is not None else default_train
    val_size = args.val_size if args.val_size is not None else default_val
    test_size = args.test_size if args.test_size is not None else default_test

    print(f"Loading {args.dataset} from HuggingFace...")
    dataset = load_dataset(args.dataset, split=config["split"])
    total = len(dataset)
    print(f"Loaded {total} samples.")

    # Shuffle deterministically
    indices = list(range(total))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Optionally take a subset (e.g. 1K) before splitting
    if args.max_samples is not None:
        if total < args.max_samples:
            print(f"Warning: dataset has {total} samples, using all (requested {args.max_samples})")
        indices = indices[: args.max_samples]

    n = len(indices)
    if n < test_size + val_size + train_size:
        raise ValueError(
            f"Not enough samples: have {n}, need at least test={test_size} + val={val_size} + train={train_size}"
        )

    # Split: test, val, train (same order as original script)
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size : test_size + val_size + train_size]

    print(f"Split sizes — train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")

    def get_normalized(idx):
        return normalize_sample(dataset[int(idx)], int(idx), config)

    # Convert and save train set (code only, no reasoning)
    train_data = [convert_to_chatml_format(get_normalized(i)) for i in train_indices]
    write_jsonl(train_data, TRAIN_OUTPUT)
    print(f"Saved {len(train_data)} train samples to {TRAIN_OUTPUT}")

    val_data = [convert_to_chatml_format(get_normalized(i)) for i in val_indices]
    write_jsonl(val_data, VAL_OUTPUT)
    print(f"Saved {len(val_data)} val samples to {VAL_OUTPUT}")

    test_data = [convert_to_test_format(get_normalized(i)) for i in test_indices]
    write_jsonl(test_data, TEST_OUTPUT)
    print(f"Saved {len(test_data)} test samples to {TEST_OUTPUT}")

    print("\nDone! Data ready for training.")

    if args.upload:
        print(f"\nUploading to HuggingFace: {args.hf_repo}")
        from datasets import Dataset

        def load_jsonl(path: str) -> list[dict]:
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        train_records = load_jsonl(TRAIN_OUTPUT)
        val_records = load_jsonl(VAL_OUTPUT)
        test_records = load_jsonl(TEST_OUTPUT)

        # Train/val: ChatML messages + extracted question/answer fields.
        def make_chatml_dataset(records: list[dict]) -> Dataset:
            ids, questions, answers, messages_col = [], [], [], []
            for i, r in enumerate(records):
                msgs = r["messages"]
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                ids.append(str(i))
                questions.append(user_msg)
                answers.append(asst_msg)
                messages_col.append(msgs)
            return Dataset.from_dict({"id": ids, "question": questions, "answer": answers, "messages": messages_col})

        # Test: raw fields + empty messages column (schema must match train/val for HF hub).
        # The messages field is unused in eval — we use llm.chat(question) directly.
        def make_test_dataset(records: list[dict]) -> Dataset:
            from datasets import Features, Sequence, Value
            features = Features({
                "id": Value("string"),
                "question": Value("string"),
                "answer": Value("string"),
                "messages": [{"content": Value("string"), "role": Value("string")}],
            })
            return Dataset.from_dict(
                {
                    "id": [str(r["id"]) for r in records],
                    "question": [r["question"] for r in records],
                    "answer": [r["answer"] for r in records],
                    "messages": [[] for _ in records],
                },
                features=features,
            )

        train_ds = make_chatml_dataset(train_records)
        val_ds = make_chatml_dataset(val_records)
        test_ds = make_test_dataset(test_records)

        # Push each split independently so train/val (with messages) and test (without) can
        # have different schemas while still landing in the same HF repository.
        print(f"  Pushing train ({len(train_ds)} samples)...")
        train_ds.push_to_hub(args.hf_repo, split="train")
        print(f"  Pushing validation ({len(val_ds)} samples)...")
        val_ds.push_to_hub(args.hf_repo, split="validation")
        print(f"  Pushing test ({len(test_ds)} samples)...")
        test_ds.push_to_hub(args.hf_repo, split="test")
        print(f"Uploaded to https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
