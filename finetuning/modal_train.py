"""
Fine-tune Qwen3-Coder-Next-Base on Modal using QLoRA (4-bit) with Unsloth.

Runs on a single B200 GPU (180GB VRAM). Uses the UIGEN dataset in ChatML
format from a Modal Volume. Logs metrics to W&B and sends an alert on completion.

Usage:
    # Quick test (5 steps)
    modal run finetuning/modal_train.py --max-steps 5

    # Full training (3 epochs)
    modal run finetuning/modal_train.py --max-steps -1
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App("uiux-finetune")

# Container image: Unsloth handles Qwen3_Next natively via its own patches.
# Do NOT override transformers — dev version (5.2.0) has a memory-hungry
# loading pipeline that OOMs. Unsloth's bundled transformers (4.57.6) works.
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[cu128-torch270]",
        "datasets",
        "hf-transfer",
        "wandb",
    )
    .env({
        "HF_HOME": "/model_cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

with train_image.imports():
    import unsloth  # noqa: F401 — must be first for patches

    import json
    import os

    import torch
    import wandb
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

# Persistent volumes
model_cache_vol = modal.Volume.from_name("uiux-model-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("uiux-training-data", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("uiux-checkpoints", create_if_missing=True)

# GPU — single H200 (141GB VRAM): Unsloth's optimized loading uses less peak
# memory than vanilla transformers+BnB. Hopper arch is fully supported.
GPU_TYPE = "H200"
TIMEOUT_HOURS = 1
MAX_RETRIES = 1

# LoRA target modules — all linear layers for best quality
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen3-Coder-Next-Base"
    max_seq_length: int = 8192
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: int = -1  # -1 means use num_epochs
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Logging
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 25

    # Experiment
    seed: int = 42
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.experiment_name = f"{model_short}-r{self.lora_r}-{timestamp}"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_training_data(tokenizer):
    """Load ChatML JSONL data from the Modal volume and format for SFTTrainer."""
    train_path = "/training_data/data/uigen_train.jsonl"
    val_path = "/training_data/data/uigen_val.jsonl"

    def read_jsonl(path):
        samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def format_samples(samples):
        """Apply the tokenizer's chat template to convert messages to text."""
        texts = []
        for sample in samples:
            text = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return Dataset.from_dict({"text": texts})

    train_samples = read_jsonl(train_path)
    print(f"Loaded {len(train_samples)} training samples")

    train_dataset = format_samples(train_samples)

    val_dataset = None
    if os.path.exists(val_path):
        val_samples = read_jsonl(val_path)
        print(f"Loaded {len(val_samples)} validation samples")
        val_dataset = format_samples(val_samples)

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_vol,
        "/training_data": data_vol,
        "/checkpoints": checkpoint_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
)
def finetune(config: TrainingConfig):
    """Run QLoRA fine-tuning on Modal with Unsloth on an H200 (141GB)."""

    # Initialize W&B
    wandb.init(
        project="uiux-train",
        name=config.experiment_name,
        config=config.__dict__,
    )
    print(f"W&B run: {wandb.run.url}")

    # Print GPU and CPU memory info
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        free_b, total_b = torch.cuda.mem_get_info(i)
        print(
            f"  GPU {i}: {name} ({mem_gb:.1f} GB total, "
            f"{free_b / 1e9:.1f} GB free)"
        )
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(
            f"  CPU RAM: {vm.total / 1e9:.1f} GB total, "
            f"{vm.available / 1e9:.1f} GB free"
        )
    except Exception:
        print("  CPU RAM: psutil not available")

    # Helper to print GPU/CPU memory at each step
    def print_mem(step_name):
        free_b, total_b = torch.cuda.mem_get_info(0)
        used_gb = (total_b - free_b) / 1e9
        free_gb = free_b / 1e9
        total_gb = total_b / 1e9
        try:
            import psutil
            cpu_free = psutil.virtual_memory().available / 1e9
            cpu_info = f" | CPU: {cpu_free:.1f} GB free"
        except Exception:
            cpu_info = ""
        print(f"  [MEM] {step_name}: GPU used {used_gb:.1f}/{total_gb:.1f} GB (free: {free_gb:.1f} GB){cpu_info}")

    print_mem("Before model load")

    # Load model with Unsloth (handles 4-bit quantization efficiently)
    # max_memory set to 200GiB to prevent accelerate from estimating bf16 size
    # (160GB) > actual GPU (141GB) and offloading to CPU. The actual 4-bit model
    # only uses ~45-60GB so it fits fine on the H200.
    print(f"Loading model: {config.model_name} (4-bit via Unsloth)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        max_memory={0: "200GiB", "cpu": "0GiB"},
    )

    print_mem("After model load")

    # Persist the downloaded model to the volume so future runs skip the download
    model_cache_vol.commit()
    print("Model cached to volume (future runs will skip download)")

    # Configure LoRA via Unsloth (handles gradient checkpointing automatically)
    print(f"Configuring LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
    )

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print_mem("After LoRA setup")

    # Load data
    print("Loading training data...")
    train_dataset, val_dataset = load_training_data(tokenizer)

    # Checkpoint directory
    checkpoint_path = f"/checkpoints/experiments/{config.experiment_name}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        output_dir=checkpoint_path,
        report_to="wandb",
        seed=config.seed,
    )

    print_mem("After data load")

    # Create trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
        args=training_args,
    )

    print(f"Training dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Experiment: {config.experiment_name}")
    print_mem("After trainer init")
    print()

    # Train
    print("Starting training...")
    trainer.train()

    print_mem("After training")

    # Save final model
    final_path = f"{checkpoint_path}/final_model"
    print(f"Saving final LoRA adapter to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Commit volume so files persist
    checkpoint_vol.commit()

    # Get final training loss from logs
    final_loss = "N/A"
    if trainer.state.log_history:
        for log in reversed(trainer.state.log_history):
            if "loss" in log:
                final_loss = f"{log['loss']:.4f}"
                break

    # Send W&B alert
    wandb.alert(
        title="Training Complete",
        text=(
            f"Model: {config.model_name}\n"
            f"Experiment: {config.experiment_name}\n"
            f"Samples: {len(train_dataset)}, Epochs: {config.num_epochs}\n"
            f"Final loss: {final_loss}\n"
            f"LoRA adapter saved to: {final_path}"
        ),
        level=wandb.AlertLevel.INFO,
    )

    wandb.finish()
    print(f"\nTraining complete! Adapter saved to {final_path}")
    return config.experiment_name


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-Coder-Next-Base",
    max_steps: int = -1,
    num_epochs: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 8192,
    experiment_name: str = None,
):
    config = TrainingConfig(
        model_name=model_name,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        experiment_name=experiment_name,
    )

    print("Launching fine-tuning on Modal (H200 141GB)")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Epochs: {config.num_epochs}, Max steps: {config.max_steps}")
    print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Experiment: {config.experiment_name}")
    print()

    experiment = finetune.remote(config)
    print(f"\nDone! Experiment: {experiment}")
    print(f"Download adapter: modal volume get uiux-checkpoints /experiments/{experiment}/final_model ./lora_adapter/")
