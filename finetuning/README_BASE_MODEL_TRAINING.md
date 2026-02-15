# Qwen2.5-Coder Base Model Training for UI/UX Code Generation

## Overview

This guide covers fine-tuning the **Qwen2.5-Coder-14B (BASE)** model for direct UI/UX code generation using the UIGEN dataset.

## Why Base Model vs Instruct Model?

| Feature | Base Model | Instruct Model |
|---------|-----------|----------------|
| **Format** | Direct text completion | ChatML conversation |
| **Tokens** | No special tokens | `<|im_start|>`, `<|im_end|>` |
| **Best For** | Code completion, direct generation | Chat, explanations, back-and-forth |
| **Training** | Simpler, full text | Requires response masking |
| **Our Use Case** | ✅ Direct prompt→code | ❌ Conversational |

## Data Format

### Original Format (ChatML)
```jsonl
{
  "messages": [
    {"role": "system", "content": "You are an expert UI/UX developer..."},
    {"role": "user", "content": "Make a job marketplace..."},
    {"role": "assistant", "content": "```html\n<!DOCTYPE html>..."}
  ]
}
```

### Converted Format (Base Model)
```jsonl
{
  "text": "# Task: Generate HTML/CSS code using Tailwind CSS\n# Requirements: Make a job marketplace...\n\n```html\n<!DOCTYPE html>..."
}
```

## Dataset

### Location
- **Original (ChatML)**: `data/uigen_*.jsonl`
- **Converted (Base)**: `data/base_format/uigen_*.jsonl`

### Statistics
- **Training**: 100 samples
- **Validation**: 10 samples
- **Test**: 80 samples

## Converting Data

If you need to re-convert or convert new data:

```bash
python finetuning/convert_uigen_to_base_format.py
```

Options:
```bash
# Custom input/output directories
python finetuning/convert_uigen_to_base_format.py \
  --input-dir data \
  --output-dir data/base_format

# Convert specific files
python finetuning/convert_uigen_to_base_format.py \
  --files uigen_train.jsonl uigen_val.jsonl
```

## Training

### Quick Test (30 steps)
```bash
modal run finetuning/unsloth/modal_qwen_code_generation.py --max-steps 30
```

### Full Training (Recommended)
```bash
modal run finetuning/unsloth/modal_qwen_code_generation.py \
  --num-epochs 3 \
  --dataset-path data/base_format/uigen_train.jsonl
```

### Custom Configuration
```bash
modal run finetuning/unsloth/modal_qwen_code_generation.py \
  --num-epochs 5 \
  --lora-r 64 \
  --lora-alpha 128 \
  --learning-rate 1e-4 \
  --max-seq-length 8192 \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

## Training Configuration

### Optimized for Code Generation
- **Model**: `Qwen/Qwen2.5-Coder-14B` (base)
- **Sequence Length**: 4096 (vs 2048 for conversational)
- **LoRA Rank**: 32 (vs 16 for conversational)
- **LR Scheduler**: Cosine (better for code tasks)
- **Gradient Accumulation**: 8 steps (for stability)

### Default Hyperparameters
```python
learning_rate: 2e-4
num_epochs: 1
batch_size: 1
gradient_accumulation_steps: 8
warmup_steps: 10
weight_decay: 0.01
lr_scheduler_type: "cosine"
lora_r: 32
lora_alpha: 32
```

## Model Inference

After training, use the base model for direct code generation:

```python
from unsloth import FastLanguageModel

# Load base model + LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen_lora",  # Your trained LoRA path
    max_seq_length=4096,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Direct completion (no chat template)
prompt = """# Task: Generate HTML/CSS code using Tailwind CSS
# Requirements: Make a modern login form with email and password fields

"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.3,  # Lower for more deterministic code
    top_p=0.9,
)
print(tokenizer.decode(outputs[0]))
```

## Key Differences from Conversational Training

| Aspect | Base Model Script | Conversational Script |
|--------|------------------|----------------------|
| **Model** | `Qwen/Qwen2.5-Coder-14B` | `unsloth/Qwen2.5-Coder-14B-Instruct` |
| **Chat Template** | ❌ None | ✅ Qwen-2.5 |
| **Response Masking** | ❌ Train on full text | ✅ Mask user/system |
| **Sequence Length** | 4096 | 2048 |
| **LoRA Rank** | 32 | 16 |
| **Data Format** | Simple text | ChatML messages |

## Files

- **Training Script**: `finetuning/unsloth/modal_qwen_code_generation.py`
- **Conversion Script**: `finetuning/convert_uigen_to_base_format.py`
- **Data Preparation**: `finetuning/prepare_uigen_data.py` (for downloading/splitting UIGEN dataset)

## Downloading Trained Models

After training completes on Modal:

```bash
# Download LoRA adapter
modal volume get qwen-checkpoints /<experiment-name>/final_model ./qwen_lora/

# Download merged 16-bit model (for production)
modal volume get qwen-checkpoints /<experiment-name>/merged_16bit ./qwen_merged/
```

## Tips for Better UI/UX Generation

1. **Increase Training Data**: 100 samples is small. Consider:
   - Increasing to 1000+ samples from full UIGEN dataset
   - Adding data augmentation (responsive variants, accessibility)
   - Mixing with other UI/UX datasets

2. **Longer Training**: Run 3-5 epochs with the full dataset

3. **Higher LoRA Rank**: Try `r=64` or `r=128` for better quality

4. **Include Reasoning**: Modify data format to include design reasoning as comments

5. **Evaluation**: Use the test set to measure code quality metrics

## Next Steps

1. ✅ Data converted to base format
2. ✅ Training script ready
3. ⏭️ Run training: `modal run finetuning/unsloth/modal_qwen_code_generation.py --max-steps 30`
4. ⏭️ Evaluate on test set
5. ⏭️ Iterate and improve based on results
