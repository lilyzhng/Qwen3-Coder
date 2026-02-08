"""
Fine-tune Qwen2.5-Coder-14B-Instruct on Modal using Unsloth (4-bit LoRA).

Based on the Unsloth Qwen2.5 Coder conversational notebook.
Runs on a single GPU (A100/H100). Uses the FineTome-100k dataset.
Logs metrics to W&B and sends an alert on completion.

Usage:
    # Quick test (30 steps)
    modal run finetuning/unsloth/modal_qwen_conversational.py --max-steps 30
    
    # Full training (1 epoch)
    modal run finetuning/unsloth/modal_qwen_conversational.py --num-epochs 1
    
    # Custom dataset size
    modal run finetuning/unsloth/modal_qwen_conversational.py --max-steps 100 --train-size 1000
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App("qwen-coder-finetune")

# Container image with Unsloth
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
    
    import torch
    import wandb
    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
model_cache_vol = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("qwen-checkpoints", create_if_missing=True)

# GPU configuration
TIMEOUT_HOURS = 6
MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "unsloth/Qwen2.5-Coder-14B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = None  # Will be set in __post_init__
    
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = 30  # Set to -1 to use num_epochs
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    
    # Data
    dataset_name: str = "mlabonne/FineTome-100k"
    train_size: int = None  # None = use full dataset
    
    # Logging
    logging_steps: int = 1
    save_steps: int = 50
    
    # Hardware
    gpu_type: str = "A100"  # A100 or H100
    
    # Experiment
    seed: int = 3407
    experiment_name: Optional[str] = None
    wandb_project: str = "qwen-coder-finetune"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.experiment_name = f"{model_short}-r{self.lora_r}-{timestamp}"


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def print_gpu_memory(step_name: str):
    """Print GPU memory stats."""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        free_memory = max_memory - used_memory
        print(f"[MEM] {step_name}: GPU = {gpu_stats.name}")
        print(f"      Used: {used_memory} GB / {max_memory} GB (Free: {free_memory} GB)")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_and_format_data(tokenizer, dataset_name: str, train_size: int = None):
    """Load dataset from HuggingFace and format for Qwen chat template."""
    print(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    if train_size:
        dataset = load_dataset(dataset_name, split=f"train[:{train_size}]")
        print(f"Loaded {len(dataset)} samples (limited to {train_size})")
    else:
        dataset = load_dataset(dataset_name, split="train")
        print(f"Loaded {len(dataset)} samples (full dataset)")
    
    # Standardize ShareGPT format to HuggingFace format
    print("Standardizing dataset format...")
    dataset = standardize_sharegpt(dataset)
    
    # Apply Qwen chat template
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            ) 
            for convo in convos
        ]
        return {"text": texts}
    
    print("Applying Qwen-2.5 chat template...")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Show example
    print(f"\nExample conversation (sample 5):")
    print("="*60)
    print(dataset[5]["text"][:500] + "..." if len(dataset[5]["text"]) > 500 else dataset[5]["text"])
    print("="*60 + "\n")
    
    return dataset


# ---------------------------------------------------------------------------
# Main Training Function - GPU-specific functions
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="A100-80GB",
    volumes={
        "/model_cache": model_cache_vol,
        "/checkpoints": checkpoint_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
)
def finetune_a100(config: TrainingConfig):
    """Run QLoRA fine-tuning with Unsloth on A100-80GB."""
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu="H100",
    volumes={
        "/model_cache": model_cache_vol,
        "/checkpoints": checkpoint_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
)
def finetune_h100(config: TrainingConfig):
    """Run QLoRA fine-tuning with Unsloth on H100."""
    return _finetune_impl(config)


# GPU function registry
_gpu_functions = {
    "A100": finetune_a100,
    "H100": finetune_h100,
}


def _finetune_impl(config: TrainingConfig):
    """Run QLoRA fine-tuning on Modal with Unsloth."""
    
    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=config.__dict__,
    )
    print(f"W&B run: {wandb.run.url}\n")
    
    print_gpu_memory("Before model load")
    
    # Load model with Unsloth (handles 4-bit quantization efficiently)
    print(f"Loading model: {config.model_name}")
    print(f"  - Max seq length: {config.max_seq_length}")
    print(f"  - 4-bit quantization: {config.load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect (Float16 for T4/V100, Bfloat16 for Ampere+)
        load_in_4bit=config.load_in_4bit,
    )
    
    print_gpu_memory("After model load")
    
    # Persist the downloaded model to the volume
    model_cache_vol.commit()
    print("Model cached to volume (future runs will skip download)\n")
    
    # Configure LoRA
    print(f"Configuring LoRA:")
    print(f"  - r: {config.lora_r}")
    print(f"  - alpha: {config.lora_alpha}")
    print(f"  - dropout: {config.lora_dropout}")
    print(f"  - target modules: {config.target_modules}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized checkpointing
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable (LoRA): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100*trainable_params/total_params,
    }, allow_val_change=True)
    
    print_gpu_memory("After LoRA setup")
    
    # Set up Qwen-2.5 chat template
    print("\nSetting up Qwen-2.5 chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )
    
    # Load and format data
    print()
    dataset = load_and_format_data(
        tokenizer,
        dataset_name=config.dataset_name,
        train_size=config.train_size,
    )
    
    print_gpu_memory("After data load")
    
    # Checkpoint directory
    checkpoint_path = f"/checkpoints/{config.experiment_name}"
    print(f"Checkpoint path: {checkpoint_path}\n")
    
    # Training arguments
    # Configure epochs vs steps properly
    if config.max_steps > 0:
        # Using max_steps mode
        num_train_epochs = 1  # Set to 1 when using max_steps
        max_steps = config.max_steps
    else:
        # Using epochs mode
        num_train_epochs = config.num_epochs
        max_steps = -1
    
    training_args = SFTConfig(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",  # Use 8-bit optimizer to save memory
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=checkpoint_path,
        report_to="wandb",
        save_steps=config.save_steps,
        save_strategy="steps",
    )
    
    # Create trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,  # Can make training 5x faster for short sequences
        args=training_args,
    )
    
    # Train only on assistant responses (mask user inputs and system prompts)
    print("Configuring response-only training (masking user/system prompts)...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    # Verify masking on a sample
    print("\nVerifying masking on sample 5:")
    print("Full text:")
    print(tokenizer.decode(trainer.train_dataset[5]["input_ids"])[:200] + "...")
    
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    masked_text = tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
    print("\nMasked (only assistant is trained):")
    print(masked_text[:200] + "...")
    print()
    
    print_gpu_memory("After trainer init")
    
    # Print training summary
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print("\n" + "="*60)
    print("Training Configuration Summary")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name} ({len(dataset)} samples)")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Training: {config.num_epochs} epoch(s), max {config.max_steps} steps")
    print(f"GPU: {config.gpu_type}")
    print(f"Experiment: {config.experiment_name}")
    print(f"W&B: {config.wandb_project}")
    print("="*60 + "\n")
    
    # Record start memory
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    
    # Train
    print("Starting training...\n")
    trainer_stats = trainer.train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Print memory and time stats
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    runtime_seconds = trainer_stats.metrics['train_runtime']
    runtime_minutes = round(runtime_seconds / 60, 2)
    
    print(f"Time: {runtime_seconds:.1f}s ({runtime_minutes} minutes)")
    print(f"Peak GPU memory: {used_memory} GB ({used_percentage}% of {max_memory} GB)")
    print(f"Memory for LoRA training: {used_memory_for_lora} GB ({lora_percentage}%)")
    
    # Get final loss
    final_loss = "N/A"
    if trainer.state.log_history:
        for log in reversed(trainer.state.log_history):
            if "loss" in log:
                final_loss = f"{log['loss']:.4f}"
                break
    print(f"Final loss: {final_loss}")
    print("="*60 + "\n")
    
    # Save final model (LoRA adapters only)
    final_path = f"{checkpoint_path}/final_model"
    print(f"Saving LoRA adapters to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Also save as merged 16-bit for inference (optional)
    merged_path = f"{checkpoint_path}/merged_16bit"
    print(f"Saving merged 16-bit model to {merged_path}...")
    try:
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit",
        )
    except Exception as e:
        print(f"Warning: Could not save merged model: {e}")
    
    # Commit checkpoint volume
    checkpoint_vol.commit()
    print("Checkpoints committed to volume")
    
    # Send W&B alert
    wandb.alert(
        title="Qwen Training Complete",
        text=(
            f"Model: {config.model_name}\n"
            f"Experiment: {config.experiment_name}\n"
            f"Dataset: {config.dataset_name} ({len(dataset)} samples)\n"
            f"Training time: {runtime_minutes} minutes\n"
            f"Final loss: {final_loss}\n"
            f"LoRA adapter saved to: {final_path}"
        ),
        level=wandb.AlertLevel.INFO,
    )
    
    wandb.finish()
    print(f"\nDone! Experiment: {config.experiment_name}")
    return config.experiment_name


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model_name: str = None,
    max_steps: int = None,
    num_epochs: int = None,
    train_size: int = None,
    lora_r: int = None,
    lora_alpha: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    max_seq_length: int = None,
    gpu_type: str = None,
    experiment_name: str = None,
):
    """
    Launch Qwen fine-tuning on Modal.
    
    Examples:
        # Quick test
        modal run finetuning/unsloth/modal_qwen_conversational.py --max-steps 30
        
        # Full epoch with custom dataset size
        modal run finetuning/unsloth/modal_qwen_conversational.py --num-epochs 1 --train-size 10000
        
        # Custom LoRA config
        modal run finetuning/unsloth/modal_qwen_conversational.py --lora-r 32 --lora-alpha 64
    """
    # Build config with CLI overrides
    config_dict = {}
    if model_name is not None:
        config_dict['model_name'] = model_name
    if max_steps is not None:
        config_dict['max_steps'] = max_steps
    if num_epochs is not None:
        config_dict['num_epochs'] = num_epochs
    if train_size is not None:
        config_dict['train_size'] = train_size
    if lora_r is not None:
        config_dict['lora_r'] = lora_r
    if lora_alpha is not None:
        config_dict['lora_alpha'] = lora_alpha
    if learning_rate is not None:
        config_dict['learning_rate'] = learning_rate
    if batch_size is not None:
        config_dict['batch_size'] = batch_size
    if gradient_accumulation_steps is not None:
        config_dict['gradient_accumulation_steps'] = gradient_accumulation_steps
    if max_seq_length is not None:
        config_dict['max_seq_length'] = max_seq_length
    if gpu_type is not None:
        config_dict['gpu_type'] = gpu_type
    if experiment_name is not None:
        config_dict['experiment_name'] = experiment_name
    
    config = TrainingConfig(**config_dict)
    
    print("="*60)
    print("Qwen2.5-Coder Fine-tuning on Modal")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"GPU: {config.gpu_type}")
    print(f"Dataset: {config.dataset_name}")
    if config.train_size:
        print(f"  - Training samples: {config.train_size}")
    else:
        print(f"  - Training samples: full dataset")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Training: {config.num_epochs} epoch(s), max {config.max_steps} steps")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Sequence length: {config.max_seq_length}")
    print(f"Experiment: {config.experiment_name}")
    print(f"W&B project: {config.wandb_project}")
    print("="*60 + "\n")
    
    # Dispatch to the appropriate GPU-specific function
    if config.gpu_type not in _gpu_functions:
        raise ValueError(f"Unknown GPU type: {config.gpu_type}. Must be A100 or H100")
    
    print(f"Launching on Modal with {config.gpu_type}...\n")
    experiment = _gpu_functions[config.gpu_type].remote(config)
    
    print("\n" + "="*60)
    print(f"Done! Experiment: {experiment}")
    print("="*60)
    print("\nTo download the LoRA adapter:")
    print(f"  modal volume get qwen-checkpoints /{experiment}/final_model ./qwen_lora/")
    print("\nTo download the merged 16-bit model:")
    print(f"  modal volume get qwen-checkpoints /{experiment}/merged_16bit ./qwen_merged/")
    print()
