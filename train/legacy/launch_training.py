"""
Launch a LoRA fine-tuning job on Together AI using BYOM.

Fine-tunes Qwen3-Coder-Next-Base using Qwen3-Next-80B-A3B-Instruct as the
base model template, with the UIGEN-T1.1-TAILWIND dataset.

Usage:
    python finetuning/launch_training.py \
        --train-file-id <file-id> \
        --val-file-id <file-id>

    # Or let the script find the most recently uploaded files automatically:
    python finetuning/launch_training.py
"""

import argparse
import os
import uuid

from dotenv import load_dotenv
from together import Together

load_dotenv()

# Model configuration
BASE_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"  # Template for infrastructure config
HF_MODEL = "Qwen/Qwen3-Coder-Next-Base"           # Actual model to fine-tune

# Training hyperparameters
N_EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 16  # Minimum allowed for this model on Together AI

# Job naming: <model-short-name>-<purpose>-<random-id>
SHORT_MODEL_NAME = "qwen3-coder-next"
PURPOSE = "uiux-mvp"

# Weights & Biases (optional, loaded from .env)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def find_uploaded_files(client: Together) -> tuple[str, str]:
    """Find the most recently uploaded train and val file IDs."""
    file_list = client.files.list()
    files = file_list.data if hasattr(file_list, "data") else list(file_list)
    train_file = None
    val_file = None

    for f in sorted(files, key=lambda x: x.created_at, reverse=True):
        if f.filename == "uigen_train.jsonl" and train_file is None:
            train_file = f.id
        elif f.filename == "uigen_val.jsonl" and val_file is None:
            val_file = f.id
        if train_file and val_file:
            break

    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(
        description="Launch LoRA fine-tuning on Together AI"
    )
    parser.add_argument(
        "--train-file-id",
        type=str,
        default=None,
        help="Together AI file ID for training data (auto-detected if not provided)",
    )
    parser.add_argument(
        "--val-file-id",
        type=str,
        default=None,
        help="Together AI file ID for validation data (auto-detected if not provided)",
    )
    args = parser.parse_args()

    client = Together()

    # Resolve file IDs
    train_file_id = args.train_file_id
    val_file_id = args.val_file_id

    if not train_file_id or not val_file_id:
        print("Auto-detecting uploaded files...")
        auto_train, auto_val = find_uploaded_files(client)

        if not train_file_id:
            train_file_id = auto_train
        if not val_file_id:
            val_file_id = auto_val

    if not train_file_id:
        print("ERROR: Could not find train file. Upload it first or pass --train-file-id")
        return
    if not val_file_id:
        print("ERROR: Could not find val file. Upload it first or pass --val-file-id")
        return

    # Generate job suffix: <model>-<purpose>-<random-id>
    random_id = uuid.uuid4().hex[:6]
    suffix = f"{SHORT_MODEL_NAME}-{PURPOSE}-{random_id}"

    print(f"Base model (template):  {BASE_MODEL}")
    print(f"HF model (fine-tune):   {HF_MODEL}")
    print(f"Train file:             {train_file_id}")
    print(f"Val file:               {val_file_id}")
    print(f"Epochs:                 {N_EPOCHS}")
    print(f"Learning rate:          {LEARNING_RATE}")
    print(f"Batch size:             {BATCH_SIZE}")
    print(f"Suffix:                 {suffix}")
    print(f"W&B logging:            {'enabled' if WANDB_API_KEY else 'disabled (set WANDB_API_KEY in .env to enable)'}")
    print()

    # Build optional kwargs
    optional_kwargs = {}
    if WANDB_API_KEY:
        optional_kwargs["wandb_api_key"] = WANDB_API_KEY

    print("Launching LoRA fine-tuning job...")
    job = client.fine_tuning.create(
        model=BASE_MODEL,
        from_hf_model=HF_MODEL,
        training_file=train_file_id,
        validation_file=val_file_id,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        suffix=suffix,
        **optional_kwargs,
    )

    print(f"\nJob launched successfully!")
    print(f"  Job ID:  {job.id}")
    print(f"  Status:  {job.status}")
    print(f"\nMonitor progress at: https://api.together.ai/fine-tuning")
    print(f"Or run: python -c \"from dotenv import load_dotenv; load_dotenv(); from together import Together; j=Together().fine_tuning.retrieve('{job.id}'); print(f'Status: {{j.status}}')\"")


if __name__ == "__main__":
    main()
