"""
Push UIGEN dataset to HuggingFace Hub.

This script takes the local JSONL files and uploads them to HuggingFace Hub
as a dataset repository.

Usage:
    python finetuning/push_to_huggingface.py --repo-name your-username/uigen-dataset
    
    # Or with custom dataset name
    python finetuning/push_to_huggingface.py --repo-name your-username/uigen-ui-code-gen
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login


def load_jsonl(file_path: str) -> list:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_dataset_dict(data_dir: str = "data/base_format") -> DatasetDict:
    """Create a DatasetDict from local JSONL files."""
    data_path = Path(data_dir)
    
    # Load train, validation, and test sets
    train_data = load_jsonl(data_path / "uigen_train.jsonl")
    val_data = load_jsonl(data_path / "uigen_val.jsonl")
    test_data = load_jsonl(data_path / "uigen_test.jsonl")
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    print(f"Loaded {len(test_data)} test samples")
    
    # Create Dataset objects
    train_dataset = Dataset.from_dict({"text": [item["text"] for item in train_data]})
    val_dataset = Dataset.from_dict({"text": [item["text"] for item in val_data]})
    test_dataset = Dataset.from_dict({"text": [item["text"] for item in test_data]})
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })
    
    return dataset_dict


def push_to_hub(
    dataset_dict: DatasetDict,
    repo_name: str,
    private: bool = False,
    token: str = None
):
    """Push dataset to HuggingFace Hub."""
    print(f"\nPushing dataset to HuggingFace Hub: {repo_name}")
    print(f"Private repository: {private}")
    
    # Push to hub
    dataset_dict.push_to_hub(
        repo_name,
        private=private,
        token=token
    )
    
    print(f"\n✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")


def create_dataset_card(repo_name: str, token: str = None):
    """Create a README.md (dataset card) for the repository."""
    
    readme_content = """---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- code-generation
- ui-design
- html-css
- tailwind
size_categories:
- n<1K
---

# UIGEN UI/UX Code Generation Dataset

This dataset contains UI/UX code generation examples formatted for training code generation models. 
Each example consists of a task description and the corresponding HTML/CSS code implementation using Tailwind CSS.

## Dataset Structure

The dataset has a single `text` column containing formatted prompts and completions:

```
# Task: Generate HTML/CSS code using Tailwind CSS
# Requirements: [specific requirements]

[HTML/CSS code implementation]
```

## Splits

- **train**: 715 examples
- **validation**: 10 examples  
- **test**: 80 examples

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("USERNAME/DATASET_NAME")

# Access training data
train_data = dataset["train"]
print(train_data[0]["text"])
```

## Use with Modal Training

```python
# In your Modal training script
from datasets import load_dataset

dataset = load_dataset("USERNAME/DATASET_NAME", split="train")
```

## Source

This dataset is derived from the UIGEN dataset, formatted specifically for code generation training with base (non-instruct) models.

## Citation

If you use this dataset, please cite the original UIGEN paper:

```bibtex
@article{uigen2023,
  title={UIGEN: A Dataset for UI Code Generation},
  author={...},
  year={2023}
}
```
"""
    
    api = HfApi(token=token)
    
    # Replace placeholders
    readme_content = readme_content.replace("USERNAME/DATASET_NAME", repo_name)
    
    print("\nCreating dataset card (README.md)...")
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )
        print("✅ Dataset card created successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not create dataset card: {e}")


def main():
    parser = argparse.ArgumentParser(description="Push UIGEN dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/base_format",
        help="Directory containing the JSONL files (default: data/base_format)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, will use cached token if not provided)"
    )
    
    args = parser.parse_args()
    
    # Login to HuggingFace (if token provided)
    if args.token:
        login(token=args.token)
    else:
        print("No token provided, using cached credentials...")
        print("If not logged in, run: huggingface-cli login")
    
    # Create dataset
    print("Creating dataset from local files...")
    dataset_dict = create_dataset_dict(args.data_dir)
    
    # Show sample
    print("\n" + "="*80)
    print("Sample from training set:")
    print("="*80)
    sample_text = dataset_dict["train"][0]["text"]
    print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
    print("="*80 + "\n")
    
    # Confirm before pushing
    response = input(f"Push dataset to {args.repo_name}? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("Aborted.")
        return
    
    # Push to hub
    push_to_hub(dataset_dict, args.repo_name, args.private, args.token)
    
    # Create dataset card
    create_dataset_card(args.repo_name, args.token)
    
    print("\n✨ All done! Your dataset is now available on HuggingFace Hub.")
    print(f"\nTo use in your Modal training script, update the config:")
    print(f"  dataset_name: str = \"{args.repo_name}\"")


if __name__ == "__main__":
    main()
