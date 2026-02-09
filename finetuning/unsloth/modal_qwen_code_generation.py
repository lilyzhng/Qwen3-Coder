"""
Fine-tune Qwen2.5-Coder-14B (base model) on Modal using Unsloth for direct code generation.

Unlike the conversational version, this uses the BASE model (not -Instruct) and trains
on direct prompt→code completion without chat templates or special tokens.

Optimized for UI/UX code generation tasks with the UIGEN dataset.

Usage:
    # Quick test (30 steps) - requires HuggingFace dataset
    modal run finetuning/unsloth/modal_qwen_code_generation.py --max-steps 30 --dataset-name lilyzhng/uigen-ui-code-gen
    
    # Full training (1 epoch) with UIGEN data from HuggingFace
    modal run finetuning/unsloth/modal_qwen_code_generation.py --num-epochs 1 --dataset-name lilyzhng/uigen-ui-code-gen
    
    # Custom training size
    modal run finetuning/unsloth/modal_qwen_code_generation.py --max-steps 100 --dataset-name lilyzhng/uigen-ui-code-gen --train-size 1000

Note: You must push your dataset to HuggingFace Hub first using:
    python finetuning/push_to_huggingface.py --repo-name lilyzhng/uigen-ui-code-gen
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App("qwen-coder-code-gen")

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
    from datasets import Dataset, load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

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
    # Model - BASE model (not instruct) for direct code generation
    model_name: str = "Qwen/Qwen2.5-Coder-14B"  # Base model, no chat template
    max_seq_length: int = 4096  # Longer for code (up from 2048)
    load_in_4bit: bool = True
    
    # LoRA
    lora_r: int = 32  # Higher rank for better code quality
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list = None  # Will be set in __post_init__
    
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = 30  # Set to -1 to use num_epochs
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Higher for stability
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"  # Cosine often better for code
    
    # Data - supports both local JSONL and HuggingFace datasets
    dataset_path: str = None  # Local JSONL path (optional, for local dev)
    dataset_name: str = None  # HuggingFace dataset name (e.g., "username/uigen-dataset")
    train_size: int = None  # None = use full dataset
    
    # Logging
    logging_steps: int = 1
    save_steps: int = 50
    
    # Hardware
    gpu_type: str = "A100"  # A100 or H100
    
    # HuggingFace Upload
    push_to_hub: bool = True  # Push to HuggingFace after training
    hf_repo_name: Optional[str] = None  # HF repo name (defaults to experiment_name)
    hf_private: bool = False  # Make repo private
    
    # Experiment
    seed: int = 3407
    experiment_name: Optional[str] = None
    wandb_project: str = "qwen-coder-code-gen"
    
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
        if self.hf_repo_name is None:
            self.hf_repo_name = self.experiment_name


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
def load_and_format_data(tokenizer, config: TrainingConfig):
    """Load dataset and format for direct code generation (no chat template)."""
    
    # Try local JSONL first, then HuggingFace
    if config.dataset_path:
        print(f"Loading local dataset: {config.dataset_path}")
        try:
            data = []
            with open(config.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            if config.train_size:
                data = data[:config.train_size]
            
            print(f"Loaded {len(data)} samples from local JSONL")
            
            # Format as prompt→code completion
            texts = []
            for item in data:
                # UIGEN format: has 'question' and 'answer' fields
                if 'messages' in item:
                    # ChatML format - extract and flatten
                    text = ""
                    for msg in item['messages']:
                        if msg['role'] == 'user':
                            text += f"# Task: {msg['content']}\n\n"
                        elif msg['role'] == 'assistant':
                            text += msg['content']
                    texts.append(text)
                elif 'question' in item and 'answer' in item:
                    # Direct question/answer format
                    prompt = f"# Task: Generate HTML/CSS code\n# Requirements: {item['question']}\n\n"
                    completion = item['answer']
                    texts.append(f"{prompt}{completion}")
                else:
                    # Generic text field
                    texts.append(item.get('text', str(item)))
            
            dataset = Dataset.from_dict({"text": texts})
            
        except FileNotFoundError:
            print(f"Warning: Local file not found: {config.dataset_path}")
            if config.dataset_name:
                print("Falling back to HuggingFace dataset...")
                return load_hf_dataset(tokenizer, config)
            else:
                raise FileNotFoundError(
                    f"Local dataset file not found: {config.dataset_path}\n"
                    f"To use a local dataset with Modal, you need to mount it using modal.Mount.\n"
                    f"Alternatively, specify a HuggingFace dataset with --dataset-name."
                )
    
    elif config.dataset_name:
        return load_hf_dataset(tokenizer, config)
    
    else:
        raise ValueError("Must specify either dataset_path or dataset_name")
    
    # Show example
    print(f"\nExample formatted text (sample 0):")
    print("="*80)
    print(dataset[0]["text"][:500] + "..." if len(dataset[0]["text"]) > 500 else dataset[0]["text"])
    print("="*80 + "\n")
    
    return dataset


def load_hf_dataset(tokenizer, config: TrainingConfig):
    """Load dataset from HuggingFace."""
    print(f"Loading HuggingFace dataset: {config.dataset_name}")
    
    if config.train_size:
        dataset = load_dataset(config.dataset_name, split=f"train[:{config.train_size}]")
    else:
        dataset = load_dataset(config.dataset_name, split="train")
    
    print(f"Loaded {len(dataset)} samples")
    
    # Check if dataset already has 'text' field in the correct format
    # If so, no need to reformat (UIGEN dataset is already formatted)
    if 'text' in dataset.column_names:
        print("Dataset already has 'text' field, using as-is")
        return dataset
    
    # Otherwise, format the data (for other dataset types)
    def formatting_prompts_func(examples):
        # When batched=True, examples is a dict with keys like {'field': [val1, val2, ...]}
        texts = []
        
        # Handle conversations format
        if 'conversations' in examples:
            for conversation in examples['conversations']:
                text = ""
                for msg in conversation:
                    text += f"{msg.get('content', msg.get('value', ''))}\n\n"
                texts.append(text)
        # Handle other formats - add more cases as needed
        else:
            # Fallback: stringify the first available field
            first_key = list(examples.keys())[0]
            texts = [str(item) for item in examples[first_key]]
        
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
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
    """Run QLoRA fine-tuning on Modal with Unsloth (BASE model, no chat template)."""
    
    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=config.__dict__,
    )
    print(f"W&B run: {wandb.run.url}\n")
    
    print_gpu_memory("Before model load")
    
    # Load BASE model (no instruct tuning, no chat template)
    print(f"Loading BASE model: {config.model_name}")
    print(f"  - Max seq length: {config.max_seq_length}")
    print(f"  - 4-bit quantization: {config.load_in_4bit}")
    print(f"  - Mode: Direct code generation (no chat template)")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
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
        use_gradient_checkpointing="unsloth",
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
    
    # NO chat template setup - base model uses direct text completion
    print("\nSkipping chat template (base model uses direct completion)")
    
    # Load and format data
    print()
    dataset = load_and_format_data(tokenizer, config)
    
    print_gpu_memory("After data load")
    
    # Checkpoint directory
    checkpoint_path = f"/checkpoints/{config.experiment_name}"
    print(f"Checkpoint path: {checkpoint_path}\n")
    
    # Training arguments
    if config.max_steps > 0:
        num_train_epochs = 1
        max_steps = config.max_steps
    else:
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
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=checkpoint_path,
        report_to="wandb",
        save_steps=config.save_steps,
        save_strategy="steps",
    )
    
    # Create trainer (NO response masking for base models)
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
        args=training_args,
    )
    
    # NO train_on_responses_only - base model trains on full text
    print("Training on full text (no masking - base model)")
    
    print_gpu_memory("After trainer init")
    
    # Print training summary
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print("\n" + "="*80)
    print("Training Configuration Summary")
    print("="*80)
    print(f"Model: {config.model_name} (BASE - no chat template)")
    print(f"Dataset: {config.dataset_path or config.dataset_name} ({len(dataset)} samples)")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LR scheduler: {config.lr_scheduler_type}")
    print(f"Training: {config.num_epochs} epoch(s), max {config.max_steps} steps")
    print(f"Sequence length: {config.max_seq_length}")
    print(f"GPU: {config.gpu_type}")
    print(f"Experiment: {config.experiment_name}")
    print(f"W&B: {config.wandb_project}")
    print("="*80 + "\n")
    
    # Record start memory
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    
    # Train
    print("Starting training...\n")
    trainer_stats = trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
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
    print("="*80 + "\n")
    
    # Push to HuggingFace (default behavior)
    if config.push_to_hub:
        import os
        print(f"\n{'='*80}")
        print(f"Pushing LoRA adapters to HuggingFace Hub...")
        print(f"Repository: {config.hf_repo_name}")
        print(f"Private: {config.hf_private}")
        print(f"{'='*80}\n")
        
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                print("⚠️  Warning: HF_TOKEN not found in environment. Skipping push to hub.")
                print("Model will only be saved to Modal volume as backup.")
                # Save to Modal as fallback
                final_path = f"{checkpoint_path}/final_model"
                print(f"\nSaving LoRA adapters to Modal volume: {final_path}...")
                model.save_pretrained(final_path)
                tokenizer.save_pretrained(final_path)
                checkpoint_vol.commit()
            else:
                # Push model and tokenizer directly to HuggingFace
                model.push_to_hub(
                    config.hf_repo_name,
                    token=hf_token,
                    private=config.hf_private,
                )
                tokenizer.push_to_hub(
                    config.hf_repo_name,
                    token=hf_token,
                    private=config.hf_private,
                )
                print(f"✅ Successfully pushed to HuggingFace: https://huggingface.co/{config.hf_repo_name}")
                print("   Model is now accessible via: AutoModel.from_pretrained('{}')".format(config.hf_repo_name))
        except Exception as e:
            print(f"⚠️  Error pushing to HuggingFace: {e}")
            print("Saving to Modal volume as fallback...")
            # Save to Modal as fallback
            final_path = f"{checkpoint_path}/final_model"
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            checkpoint_vol.commit()
            print(f"Model saved to Modal volume: {final_path}")
    else:
        # Only save to Modal if push_to_hub is disabled
        final_path = f"{checkpoint_path}/final_model"
        print(f"Saving LoRA adapters to Modal volume: {final_path}...")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        checkpoint_vol.commit()
        print("Checkpoints committed to Modal volume")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    if config.push_to_hub:
        print(f"✅ Model available at: https://huggingface.co/{config.hf_repo_name}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Training time: {runtime_minutes} minutes")
    print(f"Final loss: {final_loss}")
    print("="*80)
    
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
    dataset_path: str = None,
    dataset_name: str = None,
    experiment_name: str = None,
    push_to_hub: bool = None,
    hf_repo_name: str = None,
    hf_private: bool = None,
):
    """
    Launch Qwen BASE model fine-tuning for direct code generation on Modal.
    
    Examples:
        # Quick test with local UIGEN data (with --detach!)
        modal run --detach finetuning/unsloth/modal_qwen_code_generation.py --max-steps 30
        
        # Full epoch with HF dataset (recommended)
        modal run --detach finetuning/unsloth/modal_qwen_code_generation.py --num-epochs 1 --dataset-name lilyzhng/uigen-dataset
        
        # Disable HuggingFace push
        modal run --detach finetuning/unsloth/modal_qwen_code_generation.py --num-epochs 1 --push-to-hub false
        
        # Custom HF repo name
        modal run --detach finetuning/unsloth/modal_qwen_code_generation.py --num-epochs 1 --hf-repo-name "myuser/my-model"
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
    if dataset_path is not None:
        config_dict['dataset_path'] = dataset_path
    if dataset_name is not None:
        config_dict['dataset_name'] = dataset_name
    if experiment_name is not None:
        config_dict['experiment_name'] = experiment_name
    if push_to_hub is not None:
        config_dict['push_to_hub'] = push_to_hub
    if hf_repo_name is not None:
        config_dict['hf_repo_name'] = hf_repo_name
    if hf_private is not None:
        config_dict['hf_private'] = hf_private
    
    config = TrainingConfig(**config_dict)
    
    print("="*80)
    print("Qwen2.5-Coder BASE Model Fine-tuning (Direct Code Generation)")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"GPU: {config.gpu_type}")
    print(f"Dataset: {config.dataset_path or config.dataset_name}")
    if config.train_size:
        print(f"  - Training samples: {config.train_size}")
    else:
        print(f"  - Training samples: full dataset")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Training: {config.num_epochs} epoch(s), max {config.max_steps} steps")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LR scheduler: {config.lr_scheduler_type}")
    print(f"Sequence length: {config.max_seq_length}")
    print(f"Experiment: {config.experiment_name}")
    print(f"W&B project: {config.wandb_project}")
    print(f"Push to HuggingFace: {'Yes ✅' if config.push_to_hub else 'No'}")
    if config.push_to_hub:
        print(f"  - Repository: {config.hf_repo_name}")
        print(f"  - Private: {config.hf_private}")
    print("="*80 + "\n")
    
    # Dispatch to the appropriate GPU-specific function
    if config.gpu_type not in _gpu_functions:
        raise ValueError(f"Unknown GPU type: {config.gpu_type}. Must be A100 or H100")
    
    print(f"Launching on Modal with {config.gpu_type}...\n")
    experiment = _gpu_functions[config.gpu_type].remote(config)
    
    print("\n" + "="*80)
    print(f"Done! Experiment: {experiment}")
    print("="*80)
    print("\nTo download the LoRA adapter:")
    print(f"  modal volume get qwen-checkpoints /{experiment}/final_model ./qwen_lora/")
    print("\nTo download the merged 16-bit model:")
    print(f"  modal volume get qwen-checkpoints /{experiment}/merged_16bit ./qwen_merged/")
    print("\nFor inference with base model:")
    print("  # Load base model + LoRA adapter, then use direct completion (no chat template)")
    print()
