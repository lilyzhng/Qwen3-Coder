"""
Fine-tune Qwen3-Coder-Next-Base (80B MoE, 3B active) on Modal using ms-swift.

ms-swift handles MoE models better than UnSloth for large models:
- Proper BitsAndBytes 4-bit quantization for 80B parameter MoE models
- MoE-specific router auxiliary loss for expert load balancing
- Gradient checkpointing enabled by default
- Compatible with DeepSpeed ZeRO-2/ZeRO-3 for multi-GPU scaling

Usage:
    # Quick test (30 steps) on H100
    modal run Qwen3-Coder/ms-swift/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen --max-steps 30

    # Full training (1 epoch) on H200 for extra headroom
    modal run Qwen3-Coder/ms-swift/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen --num-epochs 1 --gpu-type H200

    # Limit dataset size
    modal run Qwen3-Coder/ms-swift/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen --max-steps 100 --train-size 1000

    # Full training (1 epoch) on B200
      modal run --detach Qwen3-Coder/ms-swift/modal_coder_base.py \
      --dataset-name lilyzhng/uigen-ui-code-gen-full \
      --max-steps -1 \
      --num-epochs 1 \
      --gpu-type B200
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App('qwen3-coder-swift')

# Container image with ms-swift
# Install from git to get Qwen3-Next support (requires ms-swift >=4.0 dev)
train_image = (
    modal.Image.from_registry('nvidia/cuda:12.8.0-devel-ubuntu22.04', add_python='3.11')
    .apt_install('git', 'build-essential')  # build-essential includes g++ and make
    .pip_install(
        'ms-swift @ git+https://github.com/modelscope/ms-swift.git',
        'transformers>=4.57,<4.58',  # ms-swift requires <4.58 (see requirements/install_all.sh)
        'trl<0.25',  # ms-swift requires <0.25
        'bitsandbytes',
        'datasets',
        'wandb',
        'hf-transfer',
        'huggingface_hub',
        'flash-linear-attention',
    )
    # causal-conv1d has no pre-built wheel for torch 2.10+, so it must compile
    # from source. --no-build-isolation lets it find the already-installed torch.
    # Set CC and CXX to force use of gcc/g++ instead of clang (which isn't installed)
    .run_commands('CC=gcc CXX=g++ pip install causal-conv1d --no-build-isolation')
    .env({
        'HF_HOME': '/root/model_cache',  # NOT /model_cache — avoid volume mount shadowing
        'HF_HUB_ENABLE_HF_TRANSFER': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'USE_HF': '1',  # Force ms-swift to use HuggingFace (not ModelScope)
    })
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Coder-Next-Base', max_workers=10)\"",
    )
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
checkpoint_vol = modal.Volume.from_name('qwen-swift-checkpoints', create_if_missing=True)

TIMEOUT_HOURS = 6


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Model — 80B MoE, 3B active params, 512 experts (10 active + 1 shared)
    model_name: str = 'Qwen/Qwen3-Coder-Next-Base'
    max_seq_length: int = 2048

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = 30  # Set to -1 to use num_epochs
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = 'cosine'

    # MoE-specific
    router_aux_loss_coef: float = 1e-3  # Router load balancing loss

    # Dataset — HuggingFace dataset with 'text' column
    dataset_name: str = None
    train_size: int = None  # None = full dataset

    # Logging
    logging_steps: int = 1
    save_steps: int = 50

    # Hardware
    gpu_type: str = 'B200'  # B200 (192GB) — needed for MoE expert conversion during loading (~140GB bf16 temporary)
    num_gpus: int = 1  # Number of GPUs (2+ enables DDP via torchrun)

    # HuggingFace Upload
    push_to_hub: bool = True
    hf_repo_name: Optional[str] = None
    hf_private: bool = False
    hf_username: str = 'lilyzhng'

    # Experiment
    seed: int = 3407
    experiment_name: Optional[str] = None
    wandb_project: str = 'qwen-coder-swift'

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_short = self.model_name.split('/')[-1]
            self.experiment_name = f'{model_short}-swift-r{self.lora_rank}-{timestamp}'
        if self.hf_repo_name is None:
            self.hf_repo_name = self.experiment_name


# ---------------------------------------------------------------------------
# GPU-specific Modal functions
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu='H100',
    cpu=8,
    volumes={
        '/checkpoints': checkpoint_vol,
    },
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune_h100(config: TrainingConfig):
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='H200',
    cpu=8,
    volumes={
        '/checkpoints': checkpoint_vol,
    },
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune_h200(config: TrainingConfig):
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='B200',
    cpu=8,
    volumes={
        '/checkpoints': checkpoint_vol,
    },
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune_b200(config: TrainingConfig):
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='B200:2',
    cpu=16,
    volumes={
        '/checkpoints': checkpoint_vol,
    },
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune_b200_2gpu(config: TrainingConfig):
    return _finetune_impl(config)


_gpu_functions = {
    'H100': finetune_h100,
    'H200': finetune_h200,
    'B200': finetune_b200,
    'B200:2': finetune_b200_2gpu,
}


# ---------------------------------------------------------------------------
# Training Implementation
# ---------------------------------------------------------------------------
def _finetune_impl(config: TrainingConfig):
    """Run QLoRA fine-tuning with ms-swift on Modal."""
    import os

    import torch

    # Print GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_gb = round(gpu.total_memory / 1024**3, 1)
        print(f'GPU: {gpu.name}, {total_gb} GB VRAM')

    # Set wandb project via env var (ms-swift/HF Trainer reads this)
    os.environ['WANDB_PROJECT'] = config.wandb_project

    # Build dataset string — ms-swift syntax: 'dataset_id#count'
    if not config.dataset_name:
        raise ValueError('Must specify --dataset-name (HuggingFace dataset with "text" column)')
    dataset_str = config.dataset_name
    if config.train_size:
        dataset_str = f'{config.dataset_name}#{config.train_size}'

    # Handle max_steps vs num_epochs
    max_steps = config.max_steps if config.max_steps > 0 else -1
    num_epochs = config.num_epochs if max_steps == -1 else 1

    output_dir = f'/checkpoints/{config.experiment_name}'

    print('\n' + '=' * 80)
    print('ms-swift QLoRA Fine-tuning — Qwen3-Coder-Next-Base (MoE)')
    print('=' * 80)
    print(f'Model: {config.model_name} (80B total, 3B active, 512 experts)')
    print(f'Quantization: BNB 4-bit NF4 (QLoRA)')
    print(f'LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}')
    print(f'Dataset: {dataset_str}')
    print(f'Training: {num_epochs} epoch(s), max {max_steps} steps')
    print(f'Batch: {config.batch_size} x {config.gradient_accumulation_steps} '
          f'= {config.batch_size * config.gradient_accumulation_steps} effective')
    print(f'Learning rate: {config.learning_rate} ({config.lr_scheduler_type})')
    print(f'MoE router aux loss coef: {config.router_aux_loss_coef}')
    print(f'Sequence length: {config.max_seq_length}')
    print(f'Output: {output_dir}')
    print(f'Experiment: {config.experiment_name}')
    print('=' * 80 + '\n')

    # -----------------------------------------------------------------------
    # Run training via ms-swift
    # -----------------------------------------------------------------------
    # swift pt = pre-training mode (base model, no chat template, full text loss)
    target_modules_str = 'q_proj k_proj v_proj o_proj gate_up_proj down_proj'

    if config.num_gpus > 1:
        # Multi-GPU: use CLI with NPROC_PER_NODE (triggers torchrun internally)
        import subprocess
        os.environ['NPROC_PER_NODE'] = str(config.num_gpus)

        cmd = [
            'swift', 'pt',
            '--model', config.model_name,
            '--dataset', dataset_str,
            '--use_hf', 'true',
            '--tuner_type', 'lora',
            '--lora_rank', str(config.lora_rank),
            '--lora_alpha', str(config.lora_alpha),
            '--target_modules', *target_modules_str.split(),
            '--quant_method', 'bnb',
            '--quant_bits', '4',
            '--bnb_4bit_compute_dtype', 'bfloat16',
            '--bnb_4bit_quant_type', 'nf4',
            '--bnb_4bit_use_double_quant', 'true',
            '--torch_dtype', 'bfloat16',
            '--max_length', str(config.max_seq_length),
            '--per_device_train_batch_size', str(config.batch_size),
            '--gradient_accumulation_steps', str(config.gradient_accumulation_steps),
            '--learning_rate', str(config.learning_rate),
            '--num_train_epochs', str(num_epochs),
            '--max_steps', str(max_steps),
            '--warmup_steps', str(config.warmup_steps),
            '--weight_decay', str(config.weight_decay),
            '--lr_scheduler_type', config.lr_scheduler_type,
            '--optim', 'adamw_8bit',
            '--router_aux_loss_coef', str(config.router_aux_loss_coef),
            '--logging_steps', str(config.logging_steps),
            '--save_steps', str(config.save_steps),
            '--save_total_limit', '2',
            '--output_dir', output_dir,
            '--report_to', 'wandb',
            '--run_name', config.experiment_name,
            '--gradient_checkpointing', 'false',
            '--seed', str(config.seed),
            '--dataloader_num_workers', '8',
        ]

        print(f'Running with {config.num_gpus} GPUs via torchrun...')
        print(f'Command: {" ".join(cmd)}\n')
        result = subprocess.run(cmd, check=True)
    else:
        # Single GPU: use Python API directly
        from swift import PretrainArguments, pretrain_main

        pretrain_main(PretrainArguments(
            model=config.model_name,
            dataset=[dataset_str],
            use_hf=True,

            # LoRA configuration
            tuner_type='lora',
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_up_proj', 'down_proj'],

            # QLoRA — 4-bit BNB quantization
            quant_method='bnb',
            quant_bits=4,
            bnb_4bit_compute_dtype='bfloat16',
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            torch_dtype='bfloat16',

            # Sequence length
            max_length=config.max_seq_length,

            # Training hyperparameters
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            optim='adamw_8bit',

            # MoE-specific
            router_aux_loss_coef=config.router_aux_loss_coef,

            # Logging & saving
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=2,
            output_dir=output_dir,
            report_to=['wandb'],
            run_name=config.experiment_name,

            # Memory optimization
            gradient_checkpointing=False,

            # Misc
            seed=config.seed,
            dataloader_num_workers=8,
        ))

    # Commit checkpoints to Modal volume
    checkpoint_vol.commit()
    print(f'\nCheckpoints saved to Modal volume: {output_dir}')

    # -----------------------------------------------------------------------
    # Push LoRA adapter to HuggingFace Hub
    # -----------------------------------------------------------------------
    if config.push_to_hub:
        hf_token = os.environ.get('HF_TOKEN')
        repo_id = f'{config.hf_username}/{config.hf_repo_name}'

        if not hf_token:
            print('Warning: HF_TOKEN not found. Skipping push to hub.')
        else:
            print(f'\nPushing LoRA adapter to HuggingFace: {repo_id}')
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                api.create_repo(repo_id, private=config.hf_private, exist_ok=True)
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=repo_id,
                    ignore_patterns=['checkpoint-*', 'runs/*', '*.bin', 'optimizer*', 'scheduler*', 'trainer_state*'],
                )
                print(f'Pushed to: https://huggingface.co/{repo_id}')
            except Exception as e:
                print(f'Warning: Failed to push to HuggingFace: {e}')
                print('LoRA adapter is still saved on the Modal volume.')

    print('\n' + '=' * 80)
    print('Training Complete!')
    print('=' * 80)
    print(f'Experiment: {config.experiment_name}')
    if config.push_to_hub:
        print(f'Model: https://huggingface.co/{config.hf_username}/{config.hf_repo_name}')
    print(f'To download from Modal: modal volume get qwen-swift-checkpoints /{config.experiment_name}/ ./output/')
    print('=' * 80)

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
    lora_rank: int = None,
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
    router_aux_loss_coef: float = None,
    num_gpus: int = None,
):
    """
    Launch Qwen3-Coder-Next-Base QLoRA fine-tuning on Modal with ms-swift.

    Examples:
        # Quick test (30 steps)
        modal run --detach Qwen3-Coder/ms-swift/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen

        # Full epoch on 2 GPUs
        modal run --detach Qwen3-Coder/ms-swift/modal_coder_base.py \\
            --dataset-name lilyzhng/uigen-ui-code-gen-full --num-epochs 1 --max-steps -1 --num-gpus 2

        # Smaller test with 500 samples
        modal run --detach Qwen3-Coder/ms-swift/modal_coder_base.py \\
            --dataset-name lilyzhng/uigen-ui-code-gen --train-size 500
    """
    config_dict = {}
    for key, val in {
        'model_name': model_name,
        'max_steps': max_steps,
        'num_epochs': num_epochs,
        'train_size': train_size,
        'lora_rank': lora_rank,
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
        'router_aux_loss_coef': router_aux_loss_coef,
        'num_gpus': num_gpus,
    }.items():
        if val is not None:
            config_dict[key] = val

    config = TrainingConfig(**config_dict)

    print('=' * 80)
    print('Qwen3-Coder-Next-Base Fine-tuning (ms-swift + QLoRA)')
    print('=' * 80)
    print(f'Model: {config.model_name}')
    print(f'GPU: {config.gpu_type} x {config.num_gpus}')
    print(f'Dataset: {config.dataset_name}')
    if config.train_size:
        print(f'  Training samples: {config.train_size}')
    else:
        print(f'  Training samples: full dataset')
    print(f'LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}')
    print(f'Batch: {config.batch_size} x {config.gradient_accumulation_steps} '
          f'= {config.batch_size * config.gradient_accumulation_steps}')
    print(f'Training: {config.num_epochs} epoch(s), max {config.max_steps} steps')
    print(f'Learning rate: {config.learning_rate} ({config.lr_scheduler_type})')
    print(f'MoE router aux loss: {config.router_aux_loss_coef}')
    print(f'Sequence length: {config.max_seq_length}')
    print(f'Experiment: {config.experiment_name}')
    print(f'Push to HuggingFace: {"Yes" if config.push_to_hub else "No"}')
    if config.push_to_hub:
        print(f'  Repository: {config.hf_username}/{config.hf_repo_name}')
    print('=' * 80 + '\n')

    # Resolve GPU function — append ':N' for multi-GPU
    gpu_key = config.gpu_type
    if config.num_gpus > 1:
        gpu_key = f'{config.gpu_type}:{config.num_gpus}'

    if gpu_key not in _gpu_functions:
        raise ValueError(f'Unknown GPU config: {gpu_key}. Available: {list(_gpu_functions.keys())}')

    print(f'Launching on Modal with {gpu_key} ({config.num_gpus} GPU(s))...\n')
    experiment = _gpu_functions[gpu_key].remote(config)

    print(f'\nDone! Experiment: {experiment}')
    print(f'To download: modal volume get qwen-swift-checkpoints /{experiment}/ ./output/')
