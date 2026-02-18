"""
Fine-tune Qwen3-Coder-Next-Base (80B MoE, 3B active) on Modal using Unsloth's MoE optimization.

Unsloth's new MoE support provides:
- 12-30x faster training vs vanilla Transformers
- >35% memory reduction
- Optimized grouped_mm backend for H100/H200

Key differences from standard Unsloth:
- NO 4-bit quantization (MoE models don't support it)
- Uses bf16 precision
- Special target_modules including gate_up_proj
- grouped_mm backend for best H100/H200 performance

Usage:
    # Quick test (30 steps) on B200 (192GB — only GPU that fits this model at bf16)
    modal run --detach Qwen3-Coder/unsloth/modal_coder_moe.py \
      --dataset-name lilyzhng/uigen-ui-code-gen --max-steps 30

    # Full training (1 epoch) on B200
    modal run --detach Qwen3-Coder/unsloth/modal_coder_moe.py \
      --dataset-name lilyzhng/uigen-ui-code-gen \
      --num-epochs 1 --max-steps -1
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App('qwen3-coder-unsloth-moe')

# Container image with Unsloth for Blackwell (B200)
# B200 requires CUDA 12.8 + Blackwell-compatible PyTorch
# See: https://unsloth.ai/docs/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth
train_image = (
    modal.Image.from_registry('nvidia/cuda:12.8.0-devel-ubuntu22.04', add_python='3.11')
    .pip_install(
        'unsloth>=2026.2.1',  # Blackwell + transformers v5 support — NOT [cu128-torch270]
        'triton>=3.3.1',  # Required for Blackwell
        'datasets',
        'hf-transfer',
        'wandb',
        'huggingface_hub',
    )
    # Upgrade to transformers v5 AFTER unsloth install.
    # Unsloth PyPI pins transformers<=4.57.6 but their docs confirm v5.1.0 works.
    # v5 stores MoE experts as nn.Parameter (not ModuleList) enabling grouped_mm — ~6x faster.
    .run_commands('pip install "transformers==5.1.0" "trl==0.27.1"')
    .env({
        'HF_HOME': '/root/model_cache',
        'HF_HUB_ENABLE_HF_TRANSFER': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'UNSLOTH_MOE_BACKEND': 'grouped_mm',  # Best for B200
    })
)

with train_image.imports():
    import os
    import unsloth  # noqa: F401 — must be first for patches

    import torch
    import wandb
    from datasets import Dataset, load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
model_cache_vol = modal.Volume.from_name('qwen-model-cache', create_if_missing=True)
checkpoint_vol = modal.Volume.from_name('qwen-unsloth-moe-checkpoints', create_if_missing=True)

TIMEOUT_HOURS = 6
MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Model - 80B MoE, 3B active params, 512 experts (10 active + 1 shared)
    model_name: str = 'Qwen/Qwen3-Coder-Next-Base'
    max_seq_length: int = 2048

    # IMPORTANT: MoE models do NOT support 4-bit quantization
    load_in_4bit: bool = False  # Must be False for MoE
    dtype: str = 'bfloat16'  # Use bf16 for better precision with MoE

    # LoRA - Unsloth MoE recommendations
    lora_r: int = 8
    lora_alpha: int = 16  # lora_r * 2
    lora_dropout: float = 0.0
    target_modules: list = None  # Will be set in __post_init__

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = 30  # Set to -1 to use num_epochs
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = 'cosine'

    # Data
    dataset_name: str = None
    train_size: int = None  # None = use full dataset

    # Logging
    logging_steps: int = 1
    save_steps: int = 50

    # Hardware
    gpu_type: str = 'B200'  # B200 (192GB) — only GPU that fits 80B MoE at bf16 (~163GB)

    # HuggingFace Upload
    push_to_hub: bool = True
    hf_repo_name: Optional[str] = None
    hf_private: bool = False
    hf_username: str = 'lilyzhng'

    # Experiment
    seed: int = 3407
    experiment_name: Optional[str] = None
    wandb_project: str = 'qwen-coder-unsloth-moe'

    def __post_init__(self):
        if self.target_modules is None:
            # MoE-specific target modules (note: gate_up_proj is fused in Qwen3)
            self.target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_up_proj', 'down_proj',  # gate_up_proj is fused in Qwen3
            ]
        if self.experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_short = self.model_name.split('/')[-1]
            self.experiment_name = f'{model_short}-unsloth-moe-r{self.lora_r}-{timestamp}'
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
        print(f'[MEM] {step_name}: GPU = {gpu_stats.name}')
        print(f'      Used: {used_memory} GB / {max_memory} GB (Free: {free_memory} GB)')


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_and_format_data(tokenizer, config: TrainingConfig):
    """Load dataset and format for direct code generation (no chat template)."""

    if not config.dataset_name:
        raise ValueError('Must specify --dataset-name (HuggingFace dataset with "text" column)')

    print(f'Loading HuggingFace dataset: {config.dataset_name}')

    if config.train_size:
        dataset = load_dataset(config.dataset_name, split=f'train[:{config.train_size}]')
    else:
        dataset = load_dataset(config.dataset_name, split='train')

    print(f'Loaded {len(dataset)} samples')

    # Check if dataset already has 'text' field
    if 'text' in dataset.column_names:
        print('Dataset already has "text" field, using as-is')
        # Show example
        print(f'\nExample text (sample 0):')
        print('='*80)
        print(dataset[0]['text'][:500] + '...' if len(dataset[0]['text']) > 500 else dataset[0]['text'])
        print('='*80 + '\n')
        return dataset

    # Otherwise, format the data
    raise ValueError(f'Dataset must have a "text" column. Found columns: {dataset.column_names}')


# ---------------------------------------------------------------------------
# GPU-specific Modal functions
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu='H100',
    volumes={
        '/model_cache': model_cache_vol,
        '/checkpoints': checkpoint_vol,
    },
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
)
def finetune_h100(config: TrainingConfig):
    """Run MoE fine-tuning with Unsloth on H100."""
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='H200',
    volumes={
        '/model_cache': model_cache_vol,
        '/checkpoints': checkpoint_vol,
    },
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
)
def finetune_h200(config: TrainingConfig):
    """Run MoE fine-tuning with Unsloth on H200."""
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='B200',
    volumes={
        '/model_cache': model_cache_vol,
        '/checkpoints': checkpoint_vol,
    },
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
)
def finetune_b200(config: TrainingConfig):
    """Run MoE fine-tuning with Unsloth on B200 (192GB — fits 80B MoE at bf16)."""
    return _finetune_impl(config)


_gpu_functions = {
    'H100': finetune_h100,
    'H200': finetune_h200,
    'B200': finetune_b200,
}


# ---------------------------------------------------------------------------
# Training Implementation
# ---------------------------------------------------------------------------
def _finetune_impl(config: TrainingConfig):
    """Run MoE fine-tuning on Modal with Unsloth (BASE model, no chat template)."""

    # Verify MoE backend
    moe_backend = os.environ.get('UNSLOTH_MOE_BACKEND', 'not set')
    print(f'Unsloth MoE backend: {moe_backend}')

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=config.__dict__,
    )
    print(f'W&B run: {wandb.run.url}\n')

    print_gpu_memory('Before model load')

    # Load BASE MoE model
    print(f'Loading BASE MoE model: {config.model_name}')
    print(f'  - Max seq length: {config.max_seq_length}')
    print(f'  - Dtype: {config.dtype} (bf16 for MoE)')
    print(f'  - 4-bit quantization: {config.load_in_4bit} (MoE does NOT support 4-bit)')
    print(f'  - MoE backend: {moe_backend}')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=getattr(torch, config.dtype) if hasattr(torch, config.dtype) else None,
        load_in_4bit=config.load_in_4bit,  # Must be False for MoE
    )

    print_gpu_memory('After model load')

    # Persist the downloaded model to the volume
    model_cache_vol.commit()
    print('Model cached to volume\n')

    # Configure LoRA with MoE-specific settings
    print(f'Configuring LoRA for MoE:')
    print(f'  - r: {config.lora_r}')
    print(f'  - alpha: {config.lora_alpha} (r * 2 for faster training)')
    print(f'  - dropout: {config.lora_dropout}')
    print(f'  - target modules: {config.target_modules}')

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel parameters:')
    print(f'  - Total: {total_params:,}')
    print(f'  - Trainable (LoRA): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')

    wandb.config.update({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_percentage': 100*trainable_params/total_params,
        'moe_backend': moe_backend,
    }, allow_val_change=True)

    print_gpu_memory('After LoRA setup')

    # NO chat template - base model uses direct text completion
    print('\nSkipping chat template (base model uses direct completion)')

    # Load and format data
    print()
    dataset = load_and_format_data(tokenizer, config)

    print_gpu_memory('After data load')

    # Checkpoint directory
    checkpoint_path = f'/checkpoints/{config.experiment_name}'
    print(f'Checkpoint path: {checkpoint_path}\n')

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
        fp16=False,  # Use bf16 for MoE
        bf16=True,
        logging_steps=config.logging_steps,
        optim='adamw_8bit',
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=checkpoint_path,
        report_to='wandb',
        save_steps=config.save_steps,
        save_strategy='steps',
    )

    # Patch fix_untrained_tokens to handle meta tensors (large MoE models offload
    # lm_head to CPU/meta via accelerate; unsloth_zoo tries .cpu() on it and crashes)
    import sys
    import unsloth_zoo.tokenizer_utils as _tok_utils
    _orig_fix_untrained = _tok_utils.fix_untrained_tokens

    def _safe_fix_untrained_tokens(*args, **kwargs):
        try:
            return _orig_fix_untrained(*args, **kwargs)
        except NotImplementedError as e:
            if 'meta tensor' in str(e):
                print('  [Skipping fix_untrained_tokens — lm_head on meta device, expected for large MoE]')
            else:
                raise

    _tok_utils.fix_untrained_tokens = _safe_fix_untrained_tokens
    # Also update the name in the compiled cache module (imported via `from ... import`)
    for _mod in sys.modules.values():
        if getattr(_mod, 'fix_untrained_tokens', None) is _orig_fix_untrained:
            _mod.fix_untrained_tokens = _safe_fix_untrained_tokens

    # Create trainer (NO response masking for base models)
    print('Initializing SFTTrainer...')
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field='text',
        max_seq_length=config.max_seq_length,
        packing=False,
        args=training_args,
    )

    print('Training on full text (no masking - base model)')

    print_gpu_memory('After trainer init')

    # Print training summary
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print('\n' + '='*80)
    print('Training Configuration Summary')
    print('='*80)
    print(f'Model: {config.model_name} (80B MoE, 3B active)')
    print(f'Framework: Unsloth with MoE optimization (backend: {moe_backend})')
    print(f'Dataset: {config.dataset_name} ({len(dataset)} samples)')
    print(f'Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {effective_batch}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Training: {num_train_epochs} epoch(s), max {max_steps} steps')
    print(f'Precision: {config.dtype} (NO 4-bit quantization for MoE)')
    print(f'Sequence length: {config.max_seq_length}')
    print(f'GPU: {config.gpu_type}')
    print(f'Experiment: {config.experiment_name}')
    print('='*80 + '\n')

    # Record start memory
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

    # Train
    print('Starting training...\n')
    trainer_stats = trainer.train()

    print('\n' + '='*80)
    print('Training Complete!')
    print('='*80)

    # Print memory and time stats
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    runtime_seconds = trainer_stats.metrics['train_runtime']
    runtime_minutes = round(runtime_seconds / 60, 2)

    print(f'Time: {runtime_seconds:.1f}s ({runtime_minutes} minutes)')
    print(f'Peak GPU memory: {used_memory} GB ({used_percentage}% of {max_memory} GB)')
    print(f'Memory for LoRA training: {used_memory_for_lora} GB ({lora_percentage}%)')

    # Get final loss
    final_loss = 'N/A'
    if trainer.state.log_history:
        for log in reversed(trainer.state.log_history):
            if 'loss' in log:
                final_loss = f"{log['loss']:.4f}"
                break
    print(f'Final loss: {final_loss}')
    print('='*80 + '\n')

    # Push to HuggingFace
    if config.push_to_hub:
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print('⚠️  Warning: HF_TOKEN not found. Saving to Modal volume only.')
            final_path = f'{checkpoint_path}/final_model'
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            checkpoint_vol.commit()
        else:
            print(f'\n{"="*80}')
            print(f'Pushing LoRA adapters to HuggingFace Hub...')
            print(f'Repository: {config.hf_username}/{config.hf_repo_name}')
            print(f'{"="*80}\n')
            try:
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
                print(f'✅ Successfully pushed to: https://huggingface.co/{config.hf_username}/{config.hf_repo_name}')
            except Exception as e:
                print(f'⚠️  Error pushing to HuggingFace: {e}')
                print('Saving to Modal volume as fallback...')
                final_path = f'{checkpoint_path}/final_model'
                model.save_pretrained(final_path)
                tokenizer.save_pretrained(final_path)
                checkpoint_vol.commit()
    else:
        final_path = f'{checkpoint_path}/final_model'
        print(f'Saving LoRA adapters to Modal volume: {final_path}...')
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        checkpoint_vol.commit()

    print('\n' + '='*80)
    print('Training Complete!')
    print('='*80)
    print(f'Experiment: {config.experiment_name}')
    print(f'Training time: {runtime_minutes} minutes')
    print(f'Final loss: {final_loss}')
    if config.push_to_hub:
        print(f'Model: https://huggingface.co/{config.hf_username}/{config.hf_repo_name}')
    print('='*80)

    wandb.finish()
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
    dataset_name: str = None,
    experiment_name: str = None,
    push_to_hub: bool = None,
    hf_repo_name: str = None,
    hf_username: str = None,
    hf_private: bool = None,
):
    """
    Launch Qwen3-Coder-Next-Base MoE fine-tuning with Unsloth on Modal.

    Examples:
        # Quick test (30 steps) on B200
        modal run --detach Qwen3-Coder/unsloth/modal_coder_moe.py \\
          --dataset-name lilyzhng/uigen-ui-code-gen

        # Full epoch on B200
        modal run --detach Qwen3-Coder/unsloth/modal_coder_moe.py \\
          --dataset-name lilyzhng/uigen-ui-code-gen \\
          --num-epochs 1 --max-steps -1
    """
    config_dict = {}
    for key, val in {
        'model_name': model_name,
        'max_steps': max_steps,
        'num_epochs': num_epochs,
        'train_size': train_size,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'max_seq_length': max_seq_length,
        'gpu_type': gpu_type,
        'dataset_name': dataset_name,
        'experiment_name': experiment_name,
        'push_to_hub': push_to_hub,
        'hf_repo_name': hf_repo_name,
        'hf_username': hf_username,
        'hf_private': hf_private,
    }.items():
        if val is not None:
            config_dict[key] = val

    config = TrainingConfig(**config_dict)

    print('='*80)
    print('Qwen3-Coder-Next-Base MoE Fine-tuning (Unsloth)')
    print('='*80)
    print(f'Model: {config.model_name} (80B MoE, 3B active)')
    print(f'GPU: {config.gpu_type}')
    print(f'Dataset: {config.dataset_name}')
    if config.train_size:
        print(f'  Training samples: {config.train_size}')
    else:
        print(f'  Training samples: full dataset')
    print(f'LoRA: r={config.lora_r}, alpha={config.lora_alpha}')
    print(f'Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}')
    print(f'Training: {config.num_epochs} epoch(s), max {config.max_steps} steps')
    print(f'Precision: {config.dtype} (NO 4-bit for MoE)')
    print(f'Sequence length: {config.max_seq_length}')
    print(f'Experiment: {config.experiment_name}')
    print(f'Push to HuggingFace: {"Yes" if config.push_to_hub else "No"}')
    if config.push_to_hub:
        print(f'  Repository: {config.hf_username}/{config.hf_repo_name}')
    print('='*80 + '\n')

    if config.gpu_type not in _gpu_functions:
        raise ValueError(f'Unknown GPU type: {config.gpu_type}. Must be H100, H200, or B200')

    print(f'Launching on Modal with {config.gpu_type}...\n')
    experiment = _gpu_functions[config.gpu_type].remote(config)

    print('\n' + '='*80)
    print(f'Done! Experiment: {experiment}')
    print('='*80)
    print('\nTo download the LoRA adapter:')
    print(f'  modal volume get qwen-unsloth-moe-checkpoints /{experiment}/final_model ./output/')
    print()
