"""
Training configuration for Qwen3-Coder fine-tuning.
Modify values here to adjust training parameters.
"""

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-Coder-Next-Base"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Training Hyperparameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_STEPS = 4  # Set to -1 to use NUM_EPOCHS instead
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
GRADIENT_CHECKPOINTING = True  # MANDATORY for 80B MoE - even H200 OOMs without it
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"

# Optimizer
OPTIMIZER = "adamw_8bit"

# Data Configuration
TRAIN_SIZE = 32  # Number of training samples to load
VAL_SIZE = 10    # Number of validation samples to load

# Logging & Checkpointing
LOGGING_STEPS = 1
SAVE_STEPS = 4
EVAL_STEPS = 2
WANDB_PROJECT = "qwen3-coder-uiux"

# UI Metrics
UI_METRICS_ENABLED = False
UI_METRICS_LOG_EVERY = 1  # Log screenshots and metrics every N steps
UI_METRICS_RENDER_SCREENSHOTS = True  # Enable Playwright visual rendering

# Hardware & Infrastructure
GPU_TYPE = "H200"  # A100, H100, or H200
TIMEOUT_HOURS = 6
MAX_RETRIES = 0

# Experiment
SEED = 42
