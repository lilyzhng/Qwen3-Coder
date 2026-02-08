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
        "numpy",  # For entropy calculations
        "playwright",  # For rendering screenshots
        "pyyaml",  # For loading config file
    )
    .run_commands(
        "playwright install --with-deps chromium"  # Install browser for screenshots
    )
    .env({
        "HF_HOME": "/model_cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

with train_image.imports():
    import unsloth  # noqa: F401 — must be first for patches

    import base64
    import json
    import os
    import re
    import yaml
    from collections import Counter
    from html.parser import HTMLParser
    from io import BytesIO
    from typing import Dict, List, Optional

    import numpy as np
    import torch
    import wandb
    from datasets import Dataset
    from playwright.sync_api import sync_playwright
    from transformers import TrainerCallback, TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    # ---------------------------------------------------------------------------
    # UI Code Metrics (SFT Monitoring)
    # ---------------------------------------------------------------------------
    
    HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <title>UI Preview</title>
</head>
<body>
{content}
</body>
</html>
"""

    def render_html_screenshot(html_code: str) -> Optional[bytes]:
        """Render HTML and capture screenshot using Playwright."""
        try:
            # Wrap code in full HTML template with Tailwind CDN
            full_html = HTML_TEMPLATE.format(content=html_code)
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                page.set_content(full_html)
                page.wait_for_timeout(1000)  # Wait for Tailwind to load
                screenshot_bytes = page.screenshot(full_page=True)
                browser.close()
                return screenshot_bytes
        except Exception as e:
            print(f"Screenshot render error: {e}")
            return None

    
    class HTMLValidator(HTMLParser):
        """Lightweight HTML validator to check syntax."""
        
        def __init__(self):
            super().__init__()
            self.errors = []
            self.tag_stack = []
            
        def handle_starttag(self, tag, attrs):
            self.tag_stack.append(tag)
            
        def handle_endtag(self, tag):
            if not self.tag_stack:
                self.errors.append(f"Unexpected closing tag: </{tag}>")
            elif self.tag_stack[-1] == tag:
                self.tag_stack.pop()
            else:
                self.errors.append(f"Tag mismatch: expected </{self.tag_stack[-1]}>, got </{tag}>")
                
        def is_valid(self):
            return len(self.errors) == 0 and len(self.tag_stack) == 0


    def extract_tailwind_classes(code: str) -> List[str]:
        """Extract Tailwind CSS classes from HTML/JSX code."""
        class_patterns = [r'className=["\'](.*?)["\']', r'class=["\'](.*?)["\']']
        classes = []
        for pattern in class_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                classes.extend(match.split())
        return classes


    def extract_color_tokens(code: str) -> List[str]:
        """Extract color-related tokens (Tailwind colors and hex codes)."""
        colors = []
        tailwind_colors = re.findall(r'\b(?:bg|text|border)-(\w+)-\d+\b', code)
        colors.extend(tailwind_colors)
        hex_colors = re.findall(r'#[0-9a-fA-F]{3,6}\b', code)
        colors.extend(hex_colors)
        return colors


    def compute_color_entropy(color_tokens: List[str]) -> float:
        """Compute entropy of color distribution. Low entropy = mode collapse."""
        if not color_tokens:
            return 0.0
        counts = Counter(color_tokens)
        total = len(color_tokens)
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy


    class UICodeMetricsCallback(TrainerCallback):
        """Custom callback to log UI-specific metrics during training."""
        
        def __init__(self, tokenizer, val_samples: Optional[List[Dict]] = None, log_every: int = 1):
            self.tokenizer = tokenizer
            self.val_samples = val_samples or []
            self.log_every = log_every
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called when trainer logs metrics."""
            if logs is None or state.global_step % self.log_every != 0:
                return
            
            # Compute perplexity from loss
            if "loss" in logs:
                perplexity = np.exp(logs["loss"])
                wandb.log({"train/perplexity": perplexity}, step=state.global_step)
            
            # Generate samples and compute UI metrics
            if self.val_samples and kwargs.get("model") is not None:
                model = kwargs["model"]
                self._log_generation_metrics(model, state.global_step)
        
        def _log_generation_metrics(self, model, step: int):
            """Generate samples and compute UI code metrics with visual rendering."""
            model.eval()
            
            all_colors = []
            all_classes = []
            valid_syntax_count = 0
            sample_generations = []
            visual_comparisons = []
            
            # Generate from first validation sample only (to save time during training)
            sample = self.val_samples[0]
            prompt_text = f"<|im_start|>user\n{sample.get('question', '')}<|im_end|>\n<|im_start|>assistant\n"
            ground_truth = sample.get("answer", "")
            
            # Generate
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Longer for better UI generation
                    temperature=0.7,
                    do_sample=True,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = generated[len(prompt_text):].strip()
            
            # Compute metrics
            validator = HTMLValidator()
            try:
                validator.feed(generated_code)
                is_valid = validator.is_valid()
            except Exception:
                is_valid = False
            valid_syntax_count += int(is_valid)
            
            colors = extract_color_tokens(generated_code)
            all_colors.extend(colors)
            
            classes = extract_tailwind_classes(generated_code)
            all_classes.extend(classes)
            
            # Render screenshots
            gen_screenshot = render_html_screenshot(generated_code)
            gt_screenshot = render_html_screenshot(ground_truth)
            
            # Create visual comparison card if screenshots rendered successfully
            if gen_screenshot and gt_screenshot:
                gen_b64 = base64.b64encode(gen_screenshot).decode('utf-8')
                gt_b64 = base64.b64encode(gt_screenshot).decode('utf-8')
                
                comparison_html = f"""
<div style="font-family: system-ui, sans-serif; width: 100%; margin: 20px 0;">
  <div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
    <div style="font-size: 14px; color: #666; margin-bottom: 4px;">Step {step}</div>
    <div style="font-size: 16px; font-weight: 600; color: #1a1a1a;">{sample.get('question', '')[:100]}...</div>
  </div>
  <div style="display: flex; gap: 16px; width: 100%;">
    <div style="flex: 1;">
      <div style="font-size: 14px; font-weight: 600; color: #e74c3c; margin-bottom: 8px;">GENERATED (Step {step})</div>
      <img src="data:image/png;base64,{gen_b64}" style="width: 100%; height: auto; border: 2px solid #e74c3c; border-radius: 6px;" />
      <div style="margin-top: 8px; font-size: 12px; color: #666;">
        ✓ Valid: {is_valid} | Colors: {len(colors)} | Classes: {len(classes)}
      </div>
    </div>
    <div style="flex: 1;">
      <div style="font-size: 14px; font-weight: 600; color: #27ae60; margin-bottom: 8px;">GROUND TRUTH</div>
      <img src="data:image/png;base64,{gt_b64}" style="width: 100%; height: auto; border: 2px solid #27ae60; border-radius: 6px;" />
    </div>
  </div>
</div>
"""
                visual_comparisons.append(comparison_html)
            
            # Save for W&B table
            sample_generations.append({
                "step": step,
                "prompt": sample.get('question', '')[:100] + "...",
                "generated_code": generated_code[:400] + "..." if len(generated_code) > 400 else generated_code,
                "syntax_valid": "✓" if is_valid else "✗",
                "num_colors": len(colors),
                "num_classes": len(classes),
            })
            
            # Aggregate metrics
            syntax_validity_rate = valid_syntax_count / 1  # Just 1 sample
            color_entropy = compute_color_entropy(all_colors)
            unique_classes = len(set(all_classes))
            
            # Log to W&B
            wandb.log({
                "eval/syntax_validity_rate": syntax_validity_rate,
                "eval/color_entropy": color_entropy,
                "eval/unique_tailwind_classes": unique_classes,
                "eval/total_colors_used": len(all_colors),
            }, step=step)
            
            # Log visual comparison as HTML
            if visual_comparisons:
                wandb.log({
                    "eval/visual_progression": wandb.Html(visual_comparisons[0])
                }, step=step)
            
            # Log sample generations as table
            if sample_generations:
                wandb.log({
                    "eval/sample_details": wandb.Table(
                        columns=list(sample_generations[0].keys()),
                        data=[list(s.values()) for s in sample_generations]
                    )
                }, step=step)
            
            model.train()


# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
model_cache_vol = modal.Volume.from_name("uiux-model-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("uiux-training-data", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("uiux-checkpoints", create_if_missing=True)

# GPU — H200 needed for training (batch_size=1, grad_accum=8, seq_len=4096)
GPU_TYPE = "A100-80GB"
TIMEOUT_HOURS = 6  # Increased: full training needs ~4-5 hours with gradient offloading
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
    max_seq_length: int = 4096
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = 4  # Set to -1 to use num_epochs
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Data
    train_size: int = 32  # Number of training samples to load
    val_size: int = 10  # Number of validation samples to load

    # Logging
    logging_steps: int = 1
    save_steps: int = 4
    eval_steps: int = 2
    
    # UI Metrics
    ui_metrics_enabled: bool = True
    ui_metrics_log_every: int = 1  # Log screenshots and UI metrics every N steps
    ui_metrics_render_screenshots: bool = True  # Enable Playwright visual rendering

    # Experiment
    seed: int = 42
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.experiment_name = f"{model_short}-r{self.lora_r}-{timestamp}"


def load_config_from_file() -> dict:
    """Load training configuration from Python config file."""
    try:
        # Import the config module
        import sys
        sys.path.insert(0, "/Users/lilyzhang/Desktop/Qwen3-Coder/finetuning")
        import train_config as cfg
        
        return {
            "model_name": cfg.MODEL_NAME,
            "max_seq_length": cfg.MAX_SEQ_LENGTH,
            "load_in_4bit": cfg.LOAD_IN_4BIT,
            "lora_r": cfg.LORA_R,
            "lora_alpha": cfg.LORA_ALPHA,
            "lora_dropout": cfg.LORA_DROPOUT,
            "learning_rate": cfg.LEARNING_RATE,
            "num_epochs": cfg.NUM_EPOCHS,
            "max_steps": cfg.MAX_STEPS,
            "batch_size": cfg.BATCH_SIZE,
            "gradient_accumulation_steps": cfg.GRADIENT_ACCUMULATION_STEPS,
            "warmup_ratio": cfg.WARMUP_RATIO,
            "weight_decay": cfg.WEIGHT_DECAY,
            "lr_scheduler_type": cfg.LR_SCHEDULER_TYPE,
            "train_size": cfg.TRAIN_SIZE,
            "val_size": cfg.VAL_SIZE,
            "logging_steps": cfg.LOGGING_STEPS,
            "save_steps": cfg.SAVE_STEPS,
            "eval_steps": cfg.EVAL_STEPS,
            "ui_metrics_enabled": cfg.UI_METRICS_ENABLED,
            "ui_metrics_log_every": cfg.UI_METRICS_LOG_EVERY,
            "ui_metrics_render_screenshots": cfg.UI_METRICS_RENDER_SCREENSHOTS,
        }
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        return {}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_training_data(tokenizer, train_size: int = 100, val_size: int = 10):
    """Load ChatML JSONL data from the Modal volume and format for SFTTrainer."""
    train_path = "/training_data/data/uigen_train.jsonl"
    val_path = "/training_data/data/uigen_val.jsonl"

    def read_jsonl(path, limit: int = None):
        samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    if limit and len(samples) >= limit:
                        break
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

    train_samples = read_jsonl(train_path, limit=train_size)
    print(f"Loaded {len(train_samples)} training samples (limit: {train_size})")

    train_dataset = format_samples(train_samples)

    val_dataset = None
    if os.path.exists(val_path):
        val_samples = read_jsonl(val_path, limit=val_size)
        print(f"Loaded {len(val_samples)} validation samples (limit: {val_size})")
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
    """Run QLoRA fine-tuning on Modal with Unsloth on A100 80GB."""

    # Initialize W&B
    wandb.init(
        project="uiux-train",
        name=config.experiment_name,
        config=config.__dict__,
    )
    print(f"W&B run: {wandb.run.url}")
    
    # Print config that was logged to W&B
    print("\n" + "="*60)
    print("📝 Config logged to W&B:")
    print("="*60)
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Training:")
    print(f"    - Learning rate: {config.learning_rate}")
    print(f"    - Epochs: {config.num_epochs}, Max steps: {config.max_steps}")
    print(f"    - Batch size: {config.batch_size}")
    print(f"    - Grad accum: {config.gradient_accumulation_steps}")
    print(f"    - Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"    - Max seq length: {config.max_seq_length}")
    print(f"  Data:")
    print(f"    - Train size: {config.train_size}")
    print(f"    - Val size: {config.val_size}")
    print(f"  Logging:")
    print(f"    - Logging steps: {config.logging_steps}")
    print(f"    - Save steps: {config.save_steps}")
    print(f"    - Eval steps: {config.eval_steps}")
    print(f"  UI Metrics:")
    print(f"    - Enabled: {config.ui_metrics_enabled}")
    print(f"    - Log every: {config.ui_metrics_log_every} steps")
    print(f"    - Render screenshots: {config.ui_metrics_render_screenshots}")
    print("="*60 + "\n")

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
    print(f"Loading model: {config.model_name} (4-bit via Unsloth)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
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
    print(f"Loading training data (train_size={config.train_size}, val_size={config.val_size})...")
    train_dataset, val_dataset = load_training_data(
        tokenizer,
        train_size=config.train_size,
        val_size=config.val_size
    )

    # Checkpoint directory
    checkpoint_path = f"/checkpoints/experiments/{config.experiment_name}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,  # Save memory by recomputing activations
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

    # Load validation samples for UI metrics callback
    val_samples_for_metrics = []
    if val_dataset:
        # Load raw validation samples (before tokenization) for generation
        val_path = "/training_data/data/uigen_val.jsonl"
        if os.path.exists(val_path):
            with open(val_path, "r") as f:
                raw_val = []
                for i, line in enumerate(f):
                    if i >= config.val_size:  # Match the val_size limit
                        break
                    line = line.strip()
                    if line:
                        raw_val.append(json.loads(line))
                # Take first 5 samples for metrics (or all if less than 5)
                val_samples_for_metrics = raw_val[:min(5, len(raw_val))]
                print(f"Loaded {len(val_samples_for_metrics)} validation samples for UI metrics")

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

    # Add UI-specific metrics callback
    if config.ui_metrics_enabled and val_samples_for_metrics:
        print(f"Adding UI code metrics callback (log_every={config.ui_metrics_log_every}, screenshots={config.ui_metrics_render_screenshots})...")
        
        ui_callback = UICodeMetricsCallback(
            tokenizer=tokenizer,
            val_samples=val_samples_for_metrics,
            log_every=config.ui_metrics_log_every
        )
        trainer.add_callback(ui_callback)
    elif not val_samples_for_metrics:
        print("⚠️  No validation samples found, skipping UI metrics callback")

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
    config_file: str = "finetuning/train_config.py",
    model_name: str = None,
    max_steps: int = None,
    num_epochs: int = None,
    lora_r: int = None,
    lora_alpha: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    max_seq_length: int = None,
    experiment_name: str = None,
):
    """
    Launch fine-tuning on Modal.
    
    Loads config from Python file, then applies CLI overrides.
    Example: modal run finetuning/modal_train.py --max-steps 5 --batch-size 2
    """
    # Load config from Python file
    print(f"Loading config from {config_file}...")
    file_config = load_config_from_file()
    
    # Start with file config, then apply CLI overrides
    config_dict = {**file_config}  # Start with file values
    
    # Override with CLI args if provided
    if model_name is not None:
        config_dict['model_name'] = model_name
    if max_steps is not None:
        config_dict['max_steps'] = max_steps
    if num_epochs is not None:
        config_dict['num_epochs'] = num_epochs
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
    if experiment_name is not None:
        config_dict['experiment_name'] = experiment_name
    
    # Create TrainingConfig from merged config
    config = TrainingConfig(**config_dict)

    print("Launching fine-tuning on Modal (A100 80GB)")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Batch size: {config.batch_size}, Grad accum: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Seq length: {config.max_seq_length}")
    print(f"  Epochs: {config.num_epochs}, Max steps: {config.max_steps}")
    print(f"  Train size: {config.train_size}, Val size: {config.val_size}")
    print(f"  UI Metrics: enabled={config.ui_metrics_enabled}, log_every={config.ui_metrics_log_every}")
    print(f"  Experiment: {config.experiment_name}")
    print()

    experiment = finetune.remote(config)
    print(f"\nDone! Experiment: {experiment}")
    print(f"Download adapter: modal volume get uiux-checkpoints /experiments/{experiment}/final_model ./lora_adapter/")
