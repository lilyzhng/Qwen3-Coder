# Quantization Workflow Guide: Qwen3-Coder-Next on Modal

Here is a comprehensive guide on how to properly quantize Qwen3-Coder-Next from bf16 to 4-bit on Modal, push it to the HuggingFace Hub, and then use the pre-quantized model for fine-tuning.

## The Challenge: VRAM

Your plan is correct and necessary. Here's why:

*   **bf16 Model Size:** The full Qwen3-Coder-Next model is ~160GB in bfloat16 precision.
*   **Single A100 VRAM:** A single A100 has 80GB of VRAM.
*   **Conclusion:** You cannot load the full bf16 model onto a single A100. You **must** use 2x A100s for the initial loading and quantization step.

Once quantized to 4-bit, the model size shrinks to ~40-46GB, which fits comfortably on a single A100 80GB for fine-tuning.

---

## Recommended Quantization Method: GPTQ

For your goal of creating a standalone, pre-quantized model that you can push to the Hub, **GPTQ is the best method.**

*   **Why not bitsandbytes?** The standard `load_in_4bit` with bitsandbytes performs quantization *on-the-fly* when you load the model. It doesn't create a saved, pre-quantized checkpoint that you can push. It's designed for the QLoRA workflow where you quantize and train in the same process.
*   **Why not AWQ?** While excellent, the AutoAWQ library (as of July 2024) does not officially support MoE models like Qwen3-Coder-Next. GPTQ, via the `gptqmodel` library, has better support for recent and complex architectures.

---

## The 2-Step Workflow on Modal

Here is the step-by-step tutorial based on HuggingFace's official documentation.

### Step 1: Quantization Job (2x A100 80GB)

Create a Python script (`quantize.py`) to run as a one-time job on Modal.

```python
import modal

# Define the Modal app and environment
image = modal.Image.debian_slim().pip_install(
    "transformers", "accelerate", "optimum", "gptqmodel", "huggingface_hub"
)
app = modal.App("quantization-job", image=image)

@app.function(
    gpu="A100:2",  # Request 2 A100 GPUs
    timeout=18000,  # 5 hours, quantization can be long
    secrets=[modal.Secret.from_name("huggingface")] # Your HF write token
)
def quantize_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    import torch

    model_id = "Qwen/Qwen3-Coder-Next-bf16" # Or the specific bf16 model name
    quantized_repo = "YourHuggingFaceUsername/Qwen3-Coder-Next-4bit-GPTQ"

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Create GPTQConfig
    # Using a common calibration dataset like "c4" is recommended
    gptq_config = GPTQConfig(
        bits=4,
        dataset="c4", # or a custom dataset: ["list of strings"]
        tokenizer=tokenizer,
        group_size=128,
    )

    # 3. Load and quantize the model
    # device_map="auto" will spread the model across the 2 GPUs
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=gptq_config,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Ensure correct dtype
    )

    # 4. Push to HuggingFace Hub
    quantized_model.push_to_hub(quantized_repo)
    tokenizer.push_to_hub(quantized_repo)

    print(f"Successfully quantized and pushed to {quantized_repo}")

```

**To run this job:**

1.  Save the code as `quantize.py`.
2.  Make sure you have a HuggingFace secret named `huggingface` in your Modal account containing a write token.
3.  Run from your terminal: `modal run quantize.py`

This job will spin up, run the quantization, push the model, and then **automatically shut down**. You only pay for the time it takes to run.

### Step 2: Fine-Tuning Job (1x A100 80GB)

Now, create a separate script (`train.py`) for your fine-tuning experiments. This script will load your newly created, pre-quantized model.

```python
import modal

# Define the Modal app and environment
image = modal.Image.debian_slim().pip_install(
    "transformers", "accelerate", "optimum", "peft", "datasets", "bitsandbytes"
)
app = modal.App("training-job", image=image)

@app.function(
    gpu="A100", # Request a single A100 GPU
    timeout=86400, # 24 hours for long training runs
    secrets=[modal.Secret.from_name("huggingface")]
)
def train_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import get_peft_model, LoraConfig

    # Load your pre-quantized model from the Hub
    model_id = "YourHuggingFaceUsername/Qwen3-Coder-Next-4bit-GPTQ"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", # Will load onto the single GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Apply LoRA adapters for fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear", # Target all linear layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # ... (Your data loading and Trainer setup here) ...

    # Example:
    # training_args = TrainingArguments(...)
    # trainer = Trainer(model=model, args=training_args, ...)
    # trainer.train()

    print("Training complete!")

```

**To run this job:**

1.  Save the code as `train.py`.
2.  Run from your terminal: `modal run train.py`

This job will load your much smaller, 4-bit model onto a single GPU, apply LoRA adapters, and begin training. Like the quantization job, it will **automatically shut down** when training is complete.

---

## References

*   [HuggingFace GPTQ Documentation](https://huggingface.co/docs/transformers/en/quantization/gptq)
*   [HuggingFace PEFT Quantization Guide](https://huggingface.co/docs/peft/main/developer_guides/quantization)
*   [Modal Unsloth Fine-tuning Example](https://modal.com/docs/examples/unsloth_finetune)
