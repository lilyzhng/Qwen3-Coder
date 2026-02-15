# UI Code Quality Metrics for SFT Training

## Overview

These custom metrics have been added to monitor **UI code generation quality** during supervised fine-tuning on the UIGEN dataset (without reasoning traces, code-only).

Based on: [SFT Monitoring Guide: UI/UX Coding Agent](./SFT%20Monitoring%20Guide_%20UI_UX%20Coding%20Agent%20(UIGEN%20Dataset).md)

## Metrics Tracked in W&B

### 1. **Perplexity** (`train/perplexity`)
- **What**: exp(loss) - how "confused" the model is
- **Target**: < 10 for well-trained UI code models
- **Formula**: `perplexity = e^loss`
- **Interpretation**:
  - High PPL (>20): Model struggling with syntax, broken code
  - Medium PPL (10-20): Acceptable but room for improvement
  - Low PPL (<10): Model is confident and accurate

### 2. **Syntax Validity Rate** (`eval/syntax_validity_rate`)
- **What**: % of generated HTML/JSX that parses correctly
- **Target**: Push toward 100% quickly (should reach 95%+ within 100 steps)
- **Computation**: HTML parser validates tag matching, no unclosed tags
- **Warning Sign**: If stuck below 90%, model may be skipping structural tokens

### 3. **Color Entropy** (`eval/color_entropy`)
- **What**: Diversity of color usage (detects "blue/purple problem")
- **Target**: Higher is better (typically 2.0-4.0 for diverse designs)
- **Formula**: Shannon entropy of color token distribution
- **Interpretation**:
  - **High entropy (>3.0)**: Good diversity, using many colors
  - **Medium entropy (2.0-3.0)**: Moderate diversity
  - **Low entropy (<1.5)**: **MODE COLLAPSE** - model defaulting to same colors (e.g., only `bg-blue-500`)

**"Blue/Purple Problem"**: If color entropy drops sharply during training, the model is reverting to generic/safe patterns instead of learning diverse UI designs.

### 4. **Unique Tailwind Classes** (`eval/unique_tailwind_classes`)
- **What**: Number of distinct Tailwind utility classes used in generations
- **Target**: Should increase over training (50+ is healthy)
- **Interpretation**:
  - Increasing: Model learning diverse layout patterns
  - Plateauing: Model may need more varied training data
  - Decreasing: Possible mode collapse

### 5. **Total Colors Used** (`eval/total_colors_used`)
- **What**: Raw count of color tokens across all generations
- **Tracks**: Absolute color usage (combines with entropy for full picture)

### 6. **Sample Generations Table** (`eval/sample_generations`)
- **What**: Actual generated code samples with metadata
- **Columns**:
  - `step`: Training step
  - `sample`: Sample index
  - `prompt`: User prompt (truncated)
  - `generated_code`: Model output (truncated)
  - `syntax_valid`: ✓ or ✗
  - `num_colors`: Color count in generation
  - `num_classes`: Tailwind class count

**Use Case**: Manual inspection to catch issues not visible in aggregate metrics.

---

## Logging Frequency

- **Training loss/perplexity**: Every step (`logging_steps=1`)
- **UI metrics (syntax, entropy, etc.)**: Every **10 steps** (`log_every=10`)
- **Sample generations**: Every 10 steps (3 samples per log)

---

## Early Warning Signs

### ⚠️ Problem: Syntax Validity Drops After Initial Improvement
- **Symptom**: Syntax rate reaches 95% at step 100, then drops to 80% at step 200
- **Cause**: Overfitting or learning rate too high
- **Fix**: Reduce LR, add regularization (weight decay), or stop training early

### ⚠️ Problem: Color Entropy Crashes
- **Symptom**: Entropy starts at 3.5, drops to 1.2 by step 300
- **Cause**: Mode collapse - model defaulting to most common patterns in base model
- **Fix**: 
  - Increase learning rate slightly (help escape local minimum)
  - Ensure data diversity (no repetitive color schemes)
  - Check if prompt masking is correct (not training on user prompts)

### ⚠️ Problem: All Samples Look Identical
- **Symptom**: Sample generations table shows similar code despite different prompts
- **Cause**: Model ignoring prompts, just outputting "average" website
- **Fix**: Verify prompt masking in SFTTrainer, ensure loss only computed on assistant response

### ⚠️ Problem: Unique Classes Plateaus Early
- **Symptom**: Unique classes hits 30 and stays there for 200+ steps
- **Cause**: Dataset has limited CSS diversity or model stuck in local optimum
- **Fix**: Check dataset for variety, consider data augmentation

---

## Expected Progression (100 samples, 3 epochs)

| Step Range | Expected Behavior |
|------------|-------------------|
| **0-50** | Perplexity drops rapidly (3.0 → 1.5), syntax validity climbs (60% → 85%) |
| **50-100** | Syntax stabilizes (90%+), color entropy settles around 2.5-3.5 |
| **100-200** | Fine-tuning quality, unique classes increase slowly |
| **200+** | Risk of overfitting - watch for validation loss divergence |

---

## How to Use These Metrics

### During Training:
1. **Monitor W&B dashboard**: Check loss, perplexity, and color entropy trends
2. **Spot-check sample generations**: Look at generated code every ~50 steps
3. **Watch for early stopping**: If validation loss diverges, stop training

### After Training:
1. **Compare metrics to baseline**: Is syntax validity higher? Color entropy maintained?
2. **Manual visual audit**: Render generated samples in browser
3. **Run full evaluation**: Use `eval_uiux.py` with LLM-as-judge + screenshots

---

## Integration

The metrics callback is automatically added to `modal_train.py` when validation samples are available:

```python
# In modal_train.py (line ~320)
ui_callback = UICodeMetricsCallback(
    tokenizer=tokenizer,
    val_samples=val_samples_for_metrics,
    log_every=10
)
trainer.add_callback(ui_callback)
```

**Requirements**:
- Validation data must exist (`/training_data/val.jsonl`)
- W&B must be initialized
- Uses first 3 validation samples for generation (to keep overhead low)

---

## Cost/Performance Impact

- **Generation overhead**: ~5-10 seconds per log (every 10 steps)
- **Total added time**: ~1-2% of training time
- **Worth it?**: Yes - early detection of mode collapse saves costly retraining

---

## References

1. SFT Monitoring Guide (this repo)
2. [UIGEN-T1.1-Qwen-14B Model Card](https://huggingface.co/Tesslate/UIGEN-T1.1-Qwen-14B)
3. [Supervised Fine-Tuning Course (HuggingFace)](https://huggingface.co/learn/llm-course/en/chapter11/3)
