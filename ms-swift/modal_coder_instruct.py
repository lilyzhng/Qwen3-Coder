"""
Fine-tune Qwen3-Coder-Next (80B MoE, instruct) on Modal using ms-swift SFT (instruction tuning).

Use this script for chat/instruction datasets (question→answer or messages). It runs
swift sft: the chat template is applied and loss is on the response (default).
For raw continuous text pre-training, use modal_coder_base.py (swift pt) instead.

ms-swift handles MoE models well: QLoRA, router aux loss, multi-GPU.

Dataset: lilyzhng/UIGEN-T1.1-split (645 train / 80 val / 80 test, split from smirki/UIGEN-T1.1-TAILWIND).
ms-swift loads the 'train' split by default. The 'test' split is reserved for eval (modal_eval_instruct.py).

Model weights are stored in a persistent Modal volume (qwen-model-cache) so they survive
image rebuilds. Download once with:
    modal run Qwen3-Coder/ms-swift/modal_coder_instruct.py::download_model

IMPORTANT: Always use --detach. The 80B MoE model takes several minutes to load, apply
LoRA, and JIT-compile the first training step. Without --detach, Modal's local heartbeat
will time out and kill the job before training even starts.

Usage:
    # One-time model download (only needed when volume is empty)
    modal run Qwen3-Coder/ms-swift/modal_coder_instruct.py::download_model

    # Standard run — 1 epoch on train split, 2× B200
    modal run --detach Qwen3-Coder/ms-swift/modal_coder_instruct.py \\
      --max-steps -1 --num-epochs 1 --gpu-type B200 --num-gpus 2

    # Quick test (30 steps)
    modal run --detach Qwen3-Coder/ms-swift/modal_coder_instruct.py --max-steps 30

    # Attention-only LoRA (compatible with vLLM runtime LoRA for MoE)
    modal run --detach Qwen3-Coder/ms-swift/modal_coder_instruct.py --max-steps 30 \\
      --target-modules "q_proj k_proj v_proj o_proj"
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App('qwen3-coder-swift-instruct')

# Official ms-swift Docker — PyTorch 2.9.0, CUDA 12.8.1, flash_attn 2.8.3, swift 3.12.5 pre-installed.
# Using this instead of NVIDIA PyTorch container solves the FSDPModule import error (needs PyTorch >=2.7)
# and avoids compiling flash_attn from source.
# US-West mirror for lowest latency from Modal (US region).
# We upgrade swift to git HEAD for Qwen3-Coder-Next support and add a few extra packages.
#
# Qwen3-Coder-Next hybrid architecture: 1 in 4 layers = full attention, 3 in 4 = linear attention.
# flash-linear-attention: pure Python PyPI wheel; causal-conv1d: pre-built cu12 wheel from GitHub.
_SWIFT_IMAGE = (
    'modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:'
    'ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5'
)
_CAUSAL_CONV1D_WHEEL = (
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/'
    'causal_conv1d-1.6.0+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
)
train_image = (
    modal.Image.from_registry(_SWIFT_IMAGE)
    .pip_install(
        'ms-swift @ git+https://github.com/modelscope/ms-swift.git',
        'wandb',
        'hf-transfer',
        'flash-linear-attention',
        _CAUSAL_CONV1D_WHEEL,
    )
    .env({
        'HF_HOME': '/model_cache',
        'HF_HUB_ENABLE_HF_TRANSFER': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'USE_HF': '1',
    })
)

checkpoint_vol = modal.Volume.from_name('qwen-swift-checkpoints', create_if_missing=True)
# Model weights volume — download once, reuse across all image rebuilds.
# Mount path matches HF_HOME so HuggingFace hub finds weights automatically.
model_vol = modal.Volume.from_name('qwen-model-cache', create_if_missing=True)
MODEL_MOUNT = '/model_cache'
TIMEOUT_HOURS = 6


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Model — 80B MoE, 3B active
    model_name: str = 'Qwen/Qwen3-Coder-Next'
    max_seq_length: int = 4096

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_up_proj', 'down_proj']
    )

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = -1  # -1 = use num_epochs; set to a positive int for a quick smoke-test
    batch_size: int = 4  # grad_checkpointing frees ~60 GB so batch_size=4 fits at seq_len=4096
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True  # required at seq_len=4096 to fit batch_size=4 on B200
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = 'cosine'

    # MoE-specific
    router_aux_loss_coef: float = 1e-3

    # Dataset — HuggingFace dataset with question/answer or messages columns.
    # Defaults to the pre-split dataset (train split only, test reserved for eval).
    dataset_name: str = 'lilyzhng/UIGEN-T1.1-split'
    train_size: int = None

    # Logging
    logging_steps: int = 1
    save_steps: int = 50

    # Hardware
    gpu_type: str = 'B200'
    num_gpus: int = 2

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
            attn_only = set(self.target_modules) == {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
            suffix = '-attn' if attn_only else ''
            self.experiment_name = f'{model_short}-sft-r{self.lora_rank}{suffix}-{timestamp}'
        if self.hf_repo_name is None:
            self.hf_repo_name = self.experiment_name


# ---------------------------------------------------------------------------
# One-time model download — run manually when volume is empty:
#   modal run Qwen3-Coder/ms-swift/modal_coder_instruct.py::download_model
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    cpu=4,
    volumes={MODEL_MOUNT: model_vol},
    secrets=[modal.Secret.from_name('hf-secret')],
    timeout=2 * 60 * 60,
)
def download_model(model_name: str = 'Qwen/Qwen3-Coder-Next'):
    """Download model weights to the persistent volume. Run once; reused by all training jobs."""
    import os
    from huggingface_hub import snapshot_download
    local_dir = os.path.join(MODEL_MOUNT, 'hub', f'models--{model_name.replace("/", "--")}')
    if os.path.exists(local_dir):
        print(f'Model already cached at {local_dir}. Nothing to do.')
    else:
        print(f'Downloading {model_name} to {MODEL_MOUNT} ...')
        snapshot_download(model_name, max_workers=10)
        model_vol.commit()
        print('Download complete and committed to volume.')


# ---------------------------------------------------------------------------
# GPU-specific Modal functions
# ---------------------------------------------------------------------------
_VOLUMES = {'/checkpoints': checkpoint_vol, MODEL_MOUNT: model_vol}

@app.function(
    image=train_image,
    gpu='H100',
    cpu=8,
    volumes=_VOLUMES,
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune_h100(config: TrainingConfig):
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='H200',
    cpu=8,
    volumes=_VOLUMES,
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune_h200(config: TrainingConfig):
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='B200',
    cpu=8,
    volumes=_VOLUMES,
    secrets=[modal.Secret.from_name('wandb-secret'), modal.Secret.from_name('hf-secret')],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def finetune_b200(config: TrainingConfig):
    return _finetune_impl(config)


@app.function(
    image=train_image,
    gpu='B200:2',
    cpu=16,
    volumes=_VOLUMES,
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
# Dataset diagnostic
# ---------------------------------------------------------------------------
def _diagnose_dataset(dataset_str: str, max_length: int, batch_size: int, num_gpus: int = 1):
    """Log dataset sample counts at each pipeline stage before training starts.

    Prints:
      1. Raw HF split sizes (how many rows the hub has)
      2. Row-by-row preprocessor run to capture the EXACT drop reason for each failed row
      3. ms-swift load_dataset output (after AutoPreprocessor)
      4. Expected steps/epoch given batch_size
    """
    import math
    from collections import Counter

    print('\n' + '─' * 80)
    print('DATASET DIAGNOSTIC (pre-training)')
    print('─' * 80)

    # ── Stage 1: raw HF dataset ─────────────────────────────────────────────
    try:
        from datasets import load_dataset as hf_load_dataset
        raw = hf_load_dataset(dataset_str.split('#')[0])
        for split_name, split_ds in raw.items():
            print(f'  HF raw  [{split_name:>12s}]: {len(split_ds):>6} rows')
        raw_train = raw['train']
        n_raw = len(raw_train)
    except Exception as e:
        print(f'  HF raw load failed: {e}')
        print('─' * 80 + '\n')
        return

    # ── Stage 2: row-by-row preprocessor to capture exact drop reasons ──────
    try:
        from swift.dataset.preprocessor.core import AutoPreprocessor
        preprocessor = AutoPreprocessor()
        # _get_preprocessor picks MessagesPreprocessor for our dataset
        row_preprocessor = preprocessor._get_preprocessor(raw_train)

        drop_reasons = Counter()
        dropped_examples = []  # store up to 3 examples for inspection

        for i, row in enumerate(raw_train):
            row = dict(row)
            try:
                result = row_preprocessor.preprocess(row)
                if result is None:
                    reason = 'preprocess() returned None'
                    drop_reasons[reason] += 1
                    if len(dropped_examples) < 3:
                        dropped_examples.append((i, reason, row))
            except Exception as e:
                reason = type(e).__name__ + ': ' + str(e)[:120]
                drop_reasons[reason] += 1
                if len(dropped_examples) < 3:
                    dropped_examples.append((i, reason, row))

        n_dropped = sum(drop_reasons.values())
        n_kept = n_raw - n_dropped

        print(f'\n  Preprocessor results ({row_preprocessor.__class__.__name__}):')
        print(f'    kept   : {n_kept:>6} / {n_raw}')
        print(f'    dropped: {n_dropped:>6} / {n_raw}')

        if n_dropped > 0:
            print(f'\n  Drop reasons:')
            for reason, count in drop_reasons.most_common():
                print(f'    [{count:>4}×] {reason}')
            print(f'\n  First dropped row examples:')
            for idx, reason, row in dropped_examples:
                msgs = row.get('messages', [])
                msg_preview = f'{len(msgs)} messages' if msgs else 'NO messages'
                content_len = sum(len(str(m.get('content', ''))) for m in msgs)
                print(f'    row {idx:>4}: {reason}')
                print(f'             messages={msg_preview}, total_content_chars={content_len}')
        else:
            print('    ✓ All rows pass the preprocessor.')

    except Exception as e:
        print(f'  Row-by-row preprocess failed: {e}')

    # ── Stage 3: ms-swift load_dataset end-to-end ───────────────────────────
    try:
        from swift.dataset import load_dataset as swift_load_dataset
        train_ds, val_ds = swift_load_dataset(
            [dataset_str],
            use_hf=True,
            split_dataset_ratio=0.0,
            num_proc=4,
            load_from_cache_file=False,
        )
        n_train = len(train_ds) if train_ds is not None else 0
        print(f'\n  swift load_dataset [train]: {n_train:>6} rows')

        global_batch = batch_size * num_gpus
        expected_steps = math.ceil(n_train / global_batch)
        print(f'\n  max_length        : {max_length}')
        print(f'  batch_size/gpu    : {batch_size}')
        print(f'  num_gpus          : {num_gpus}')
        print(f'  global batch size : {global_batch}')
        print(f'  expect steps      : ~{expected_steps} / epoch')

    except Exception as e:
        print(f'  swift load_dataset failed: {e}')

    print('─' * 80 + '\n')


# ---------------------------------------------------------------------------
# Training Implementation (swift sft)
# ---------------------------------------------------------------------------
def _finetune_impl(config: TrainingConfig):
    """Run QLoRA SFT (instruction tuning) with ms-swift on Modal."""
    import os

    import torch

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_gb = round(gpu.total_memory / 1024**3, 1)
        print(f'GPU: {gpu.name}, {total_gb} GB VRAM')

    os.environ['WANDB_PROJECT'] = config.wandb_project

    dataset_str = config.dataset_name
    if config.train_size:
        dataset_str = f'{config.dataset_name}#{config.train_size}'

    max_steps = config.max_steps if config.max_steps > 0 else -1
    num_epochs = config.num_epochs if max_steps == -1 else 1
    output_dir = f'/checkpoints/{config.experiment_name}'

    print('\n' + '=' * 80)
    print('ms-swift SFT (instruction tuning) QLoRA — Qwen3-Coder-Next (MoE)')
    print('=' * 80)
    print(f'Model: {config.model_name} (80B total, 3B active, 512 experts)')
    print('Quantization: BNB 4-bit NF4 (QLoRA)')
    print(f'LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}')
    print(f'LoRA targets: {" ".join(config.target_modules)}')
    print(f'Dataset: {dataset_str}')
    print(f'Training: {num_epochs} epoch(s), max {max_steps} steps')
    print(f'Batch: {config.batch_size} x {config.gradient_accumulation_steps} '
          f'= {config.batch_size * config.gradient_accumulation_steps} effective')
    print(f'Learning rate: {config.learning_rate} ({config.lr_scheduler_type})')
    print(f'MoE router aux loss coef: {config.router_aux_loss_coef}')
    print(f'Sequence length: {config.max_seq_length}')
    print(f'Gradient checkpointing: {config.gradient_checkpointing}')
    print(f'Attention: flash_attn')
    print(f'Output: {output_dir}')
    print(f'Experiment: {config.experiment_name}')
    print('=' * 80 + '\n')

    _diagnose_dataset(dataset_str, config.max_seq_length, config.batch_size, config.num_gpus)

    if config.num_gpus > 1:
        import subprocess
        os.environ['NPROC_PER_NODE'] = str(config.num_gpus)

        cmd = [
            'swift', 'sft',
            '--model', config.model_name,
            '--dataset', dataset_str,
            '--use_hf', 'true',
            '--tuner_type', 'lora',
            '--lora_rank', str(config.lora_rank),
            '--lora_alpha', str(config.lora_alpha),
            '--target_modules', *config.target_modules,
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
            '--gradient_checkpointing', str(config.gradient_checkpointing).lower(),
            '--gradient_checkpointing_kwargs', '{"determinism_check": "none"}',
            '--attn_impl', 'flash_attn',
            '--seed', str(config.seed),
            '--dataloader_num_workers', '8',
            '--load_from_cache_file', 'false',  # always re-preprocess; avoids stale cached dataset
        ]

        print(f'Running with {config.num_gpus} GPUs via torchrun...')
        print('Command: ' + ' '.join(cmd) + '\n')
        subprocess.run(cmd, check=True)
    else:
        from swift import SftArguments, sft_main

        sft_main(SftArguments(
            model=config.model_name,
            dataset=[dataset_str],
            use_hf=True,
            load_from_cache_file=False,  # always re-preprocess; avoids stale cached dataset

            tuner_type='lora',
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,

            quant_method='bnb',
            quant_bits=4,
            bnb_4bit_compute_dtype='bfloat16',
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            torch_dtype='bfloat16',

            max_length=config.max_seq_length,

            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            optim='adamw_8bit',

            router_aux_loss_coef=config.router_aux_loss_coef,

            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=2,
            output_dir=output_dir,
            report_to=['wandb'],
            run_name=config.experiment_name,

            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={'determinism_check': 'none'},
            attn_impl='flash_attn',

            seed=config.seed,
            dataloader_num_workers=8,
        ))

    checkpoint_vol.commit()
    print(f'\nCheckpoints saved to Modal volume: {output_dir}')

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
    target_modules: str = None,
):
    """
    Launch Qwen3-Coder-Next SFT (instruction tuning) on Modal with ms-swift.

    Use for chat datasets (question/answer or messages). For raw text pre-training use modal_coder_base.py.

    Always use --detach (80B MoE takes minutes to load; without it, heartbeat timeout kills the job).

    Default dataset: lilyzhng/UIGEN-T1.1-split (train split, 645 samples).
    Pass --dataset-name to override with a different HF dataset.

    Examples:
        # Standard 1-epoch run on default dataset, 2× B200
        modal run --detach Qwen3-Coder/ms-swift/modal_coder_instruct.py \\
            --num-epochs 1 --max-steps -1 --gpu-type B200 --num-gpus 2

        # Quick test (30 steps)
        modal run --detach Qwen3-Coder/ms-swift/modal_coder_instruct.py --max-steps 30
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

    if target_modules is not None:
        config_dict['target_modules'] = target_modules.split()

    config = TrainingConfig(**config_dict)

    print('=' * 80)
    print('Qwen3-Coder-Next SFT (ms-swift + QLoRA)')
    print('=' * 80)
    print(f'Model: {config.model_name}')
    print(f'GPU: {config.gpu_type} x {config.num_gpus}')
    print(f'Dataset: {config.dataset_name}')
    if config.train_size:
        print(f'  Training samples: {config.train_size}')
    else:
        print('  Training samples: full dataset')
    print(f'LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}')
    print(f'LoRA targets: {" ".join(config.target_modules)}')
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

    gpu_key = config.gpu_type
    if config.num_gpus > 1:
        gpu_key = f'{config.gpu_type}:{config.num_gpus}'

    if gpu_key not in _gpu_functions:
        raise ValueError(f'Unknown GPU config: {gpu_key}. Available: {list(_gpu_functions.keys())}')

    print(f'Launching on Modal with {gpu_key} ({config.num_gpus} GPU(s))...\n')
    experiment = _gpu_functions[gpu_key].remote(config)

    print(f'\nDone! Experiment: {experiment}')
    print(f'To download: modal volume get qwen-swift-checkpoints /{experiment}/ ./output/')
