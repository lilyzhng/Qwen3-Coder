# Pushing UIGEN Dataset to HuggingFace Hub

This guide walks you through pushing your UIGEN dataset to HuggingFace Hub so you can use it with Modal training.

## Prerequisites

1. **HuggingFace Account**: Create one at [huggingface.co](https://huggingface.co/join) if you don't have one
2. **HuggingFace Token**: Get your access token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Install Dependencies**: Already installed in your environment

## Step 1: Login to HuggingFace

```bash
# Login using the CLI (easiest method)
huggingface-cli login

# Or set your token as an environment variable
export HF_TOKEN=your_token_here
```

## Step 2: Push Dataset to HuggingFace Hub

```bash
# Replace 'your-username' with your actual HuggingFace username
python finetuning/push_to_huggingface.py --repo-name your-username/uigen-ui-code-gen

# For a private dataset (recommended if you don't want it publicly visible)
python finetuning/push_to_huggingface.py --repo-name your-username/uigen-ui-code-gen --private

# With explicit token (if not logged in via CLI)
python finetuning/push_to_huggingface.py \
    --repo-name your-username/uigen-ui-code-gen \
    --token your_hf_token_here
```

The script will:
- Load your local JSONL files from `data/base_format/`
- Show you a sample of the data
- Ask for confirmation before pushing
- Upload to HuggingFace Hub
- Create a dataset card (README.md)

## Step 3: Verify Upload

Visit `https://huggingface.co/datasets/your-username/uigen-ui-code-gen` to verify your dataset was uploaded successfully.

## Step 4: Update Modal Training Script

Once your dataset is on HuggingFace, you can train on Modal:

```bash
# Test with 30 steps
modal run finetuning/unsloth/modal_qwen_code_generation.py \
    --max-steps 30 \
    --dataset-name your-username/uigen-ui-code-gen

# Full training (1 epoch)
modal run finetuning/unsloth/modal_qwen_code_generation.py \
    --num-epochs 1 \
    --dataset-name your-username/uigen-ui-code-gen

# With custom training size
modal run finetuning/unsloth/modal_qwen_code_generation.py \
    --max-steps 100 \
    --dataset-name your-username/uigen-ui-code-gen \
    --train-size 1000
```

## Benefits of Using HuggingFace Hub

✅ **No file mounting issues** - Modal downloads directly from HuggingFace  
✅ **Faster iteration** - No need to rebuild container images when data changes  
✅ **Version control** - Track dataset versions and changes  
✅ **Easy sharing** - Share datasets with team members  
✅ **Reproducibility** - Anyone can use the exact same dataset  

## Troubleshooting

### "Could not find dataset"
- Make sure you're using the correct dataset name format: `username/dataset-name`
- Verify the dataset exists at `https://huggingface.co/datasets/your-username/dataset-name`

### Authentication errors
- Run `huggingface-cli login` again
- Make sure your HuggingFace secret is set up in Modal: [Modal Secrets](https://modal.com/secrets)

### Dataset is private but Modal can't access it
- Make sure your `hf-secret` in Modal contains a valid HuggingFace token with read permissions
- Alternatively, make the dataset public (remove `--private` flag)

## Example Output

```
$ python finetuning/push_to_huggingface.py --repo-name lily-ai/uigen-ui-code-gen

Creating dataset from local files...
Loaded 101 training samples
Loaded 11 validation samples
Loaded 81 test samples

================================================================================
Sample from training set:
================================================================================
# Task: Generate HTML/CSS code using Tailwind CSS
# Requirements: Make a freelance job marketplace with responsive card layouts...
================================================================================

Push dataset to lily-ai/uigen-ui-code-gen? (yes/no): yes

Pushing dataset to HuggingFace Hub: lily-ai/uigen-ui-code-gen
Private repository: False

✅ Dataset successfully pushed to: https://huggingface.co/datasets/lily-ai/uigen-ui-code-gen

Creating dataset card (README.md)...
✅ Dataset card created successfully

✨ All done! Your dataset is now available on HuggingFace Hub.

To use in your Modal training script, update the config:
  dataset_name: str = "lily-ai/uigen-ui-code-gen"
```
