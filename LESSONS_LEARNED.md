# Lessons Learned - Qwen3-Coder Fine-tuning

## Modal Job Management

### Issue: Local Client Disconnects During Long-Running Jobs

**Problem:**
When launching Modal jobs from a local terminal using `modal run`, the local client maintains a heartbeat connection. If the job runs longer than expected or involves long operations (like model merging), the heartbeat can fail, causing the client to disconnect and terminate the job prematurely.

**Example Error:**
```
[modal-client] Loop attempt for _run_app.<locals>.heartbeat failed
Stopping app - local client disconnected
Runner terminated.
```

**Solution:**
Always use the `--detach` flag when running long jobs from your local machine:

```bash
# ❌ BAD - Client stays connected, can timeout
modal run unsloth/modal_coder_base.py --num-epochs 1 --dataset-name lilyzhng/uigen-ui-code-gen-full

# ✅ GOOD - Job continues even if local client disconnects
modal run --detach unsloth/modal_coder_base.py --num-epochs 1 --dataset-name lilyzhng/uigen-ui-code-gen-full --max-steps -1
```

**When to Use `--detach`:**
- Training jobs (always, even for quick experiments)
- Any job with model merging operations
- Jobs that might take longer than 5-10 minutes
- When your local machine might sleep or lose connection

**Benefits:**
- Job continues running on Modal even if your laptop sleeps
- No risk of heartbeat timeout
- Can close terminal/disconnect from network
- View progress via Modal dashboard: `https://modal.com/apps`

**Note:** You can still monitor detached jobs by checking the Modal web UI or using `modal app logs <app-id>`.

---

## Model Merging Considerations

### Decision: Skip Merged Model Saving (LoRA Adapters Only)

**What Changed:**
Training scripts now **only save LoRA adapters** and skip the slow merged 16-bit model saving step that was causing timeouts.

**Why:**
- Merging 14B models takes 2-3 minutes and caused Modal client disconnects
- LoRA adapters are sufficient for most use cases
- Merged models can be created later if needed using a separate job
- Faster training completion and more reliable uploads

**What Gets Saved:**
- ✅ **LoRA adapters only** (pushed to HuggingFace)
- ❌ ~~Merged 16-bit model~~ (removed to avoid timeouts)

**How to Use the LoRA Adapters:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B")
model = PeftModel.from_pretrained(base_model, "your-username/your-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-lora-adapter")

# Or use directly with HF pipeline
from transformers import pipeline
pipe = pipeline("text-generation", model="your-username/your-lora-adapter")
```

**If You Need a Merged Model:**
Create a separate merge job or use the Unsloth merge script after downloading the adapters.

---

## Model Storage: Modal Volumes vs HuggingFace

### Best Practice: Push Directly to HuggingFace (Default Behavior)

**Current Implementation:**
Both training scripts now **default to pushing models directly to HuggingFace** after training completes. Modal volumes are only used as a fallback if HuggingFace push fails.

**Why HuggingFace is Better:**
- ✅ Easy to load anywhere: `AutoModel.from_pretrained("your-username/model-name")`
- ✅ Free hosting for public models
- ✅ Built-in versioning (git-based)
- ✅ Model cards for documentation
- ✅ Can share with others or use in other projects
- ✅ Works with all HF tools (inference endpoints, spaces, etc.)
- ✅ Integrates with deployment platforms
- ✅ Automatic model card generation with training details

**Modal Volumes Limitations:**
- ❌ Harder to access (requires Modal CLI or API)
- ❌ Not shareable/public
- ❌ No version control or model cards
- ❌ Costs storage fees on Modal
- ❌ Not portable (locked to Modal infrastructure)
- ❌ Can't easily load for inference elsewhere

**How It Works:**
1. Training completes on Modal (with GPU)
2. **Model automatically pushes to HuggingFace** (`push_to_hub=True` by default)
3. If HuggingFace push fails → Falls back to Modal volume as backup
4. Load from HuggingFace for inference/deployment

**To Disable HuggingFace Push:**
```bash
modal run --detach unsloth/modal_coder_base.py \
  --num-epochs 1 --max-steps -1 \
  --push-to-hub false
```

**Custom Repository Name:**
```bash
modal run --detach unsloth/modal_coder_base.py \
  --num-epochs 1 --max-steps -1 \
  --hf-repo-name "myusername/my-custom-model-name"
```

**Note:** Make sure HuggingFace token is configured in Modal secrets:
```bash
modal secret create hf-secret HF_TOKEN=your_huggingface_token
```

---

## Bug Fixes

### W&B Alert Error (Fixed)

**Issue:**
Earlier versions had a bug where `wandb.alert()` was called after `wandb.finish()`, causing the training to crash with:
```
wandb.errors.errors.UsageError: Run is finished. The call to `alert` will be ignored.
```

**Fix:**
Removed W&B alerts entirely. Training summary is now printed to console instead. W&B dashboard still shows all metrics and training progress.

**Impact:**
- No more crashes at the end of training
- Cleaner, more reliable completion
- All training metrics still logged to W&B

---

## Base Model vs Instruct Model: Prompt Format Matters

### Issue: Model Generates Instructions Instead of Code

**Problem:**
When evaluating a fine-tuned LoRA model on the base `Qwen/Qwen2.5-Coder-14B`, the model generated descriptive text and instructions instead of actual HTML/CSS code.

**Example of Wrong Output:**
```
The portal should have a modern design with intuitive navigation and user-friendly features. 
The employee directories should include contact information, job titles, and department affiliations...

To create a corporate intranet portal with the features you've described, we'll need to follow these steps:
1. Define Requirements...
2. Design Layout...
```

**Expected Output:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <!-- Actual UI code -->
</body>
</html>
```

### Root Cause: Prompt Format Mismatch

The model was trained on data with a specific prompt format:
```
# Task: Generate HTML/CSS code using Tailwind CSS
# Requirements: Make a productivity timer app with minimalistic design...

```html
<!DOCTYPE html>
...
```

But during evaluation, only the raw requirements were sent:
```
Make a productivity timer app with minimalistic design...
```

**Base models are completion models** - they predict the next tokens based on the input. When given unfamiliar text format, they continue naturally as text rather than generating code.

### Solution: Match Training Prompt Format Exactly

**Training Data Format:**
```python
PROMPT_TEMPLATE = "# Task: Generate HTML/CSS code using Tailwind CSS\n# Requirements: {requirements}\n\n"
```

**Evaluation Must Use Same Format:**
```python
# ❌ WRONG - Raw requirements only
prompt = "Make a productivity timer app..."

# ✅ CORRECT - Full prompt matching training format
prompt = """# Task: Generate HTML/CSS code using Tailwind CSS
# Requirements: Make a productivity timer app...

"""
```

### Key Differences: Base vs Instruct Models

| Aspect | Base Model | Instruct Model |
|--------|------------|----------------|
| **Purpose** | Text completion | Instruction following |
| **Prompt sensitivity** | Very sensitive - needs exact format | More robust to variations |
| **Training required** | More data/epochs for new patterns | Already understands task structure |
| **Use case** | Domain-specific completion | General instruction following |

### Recommendations

1. **When using base models for fine-tuning:**
   - Use a consistent, distinctive prompt format in training data
   - Always use the EXACT same prompt format during inference
   - Consider adding clear delimiters (e.g., `# Task:`, `# Requirements:`)

2. **Consider using Instruct models instead:**
   - `Qwen/Qwen2.5-Coder-14B-Instruct` already understands "generate X" instructions
   - More forgiving of prompt variations
   - May require less training data to adapt

3. **Verify prompt format before evaluation:**
   - Print the actual prompt being sent to the model
   - Compare against training data format
   - Test with a single sample first

### Files Changed to Fix This Issue

- `finetuning/modal_eval.py` - Added `PROMPT_TEMPLATE` constant and updated `load_test_data()` to wrap requirements in full prompt format
- `unsloth/coder_colab.ipynb` - Updated test cells to use training prompt format

---

## Base Model Training Commands

### Quick Reference

```bash
# 1-step sanity check (run before full training)
modal run unsloth/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen-full --max-steps 1

# Full 1 epoch over all 715 samples (ALWAYS use --detach for long jobs)
modal run --detach unsloth/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen-full --num-epochs 1 --max-steps -1
```

### Important: max-steps vs num-epochs

By default, `max_steps=30` **overrides** `num_epochs`. With `batch_size=1` × `gradient_accumulation_steps=8`:

- **30 steps (default)** = 30 × 8 = **240 samples** — does NOT use full dataset
- **Full epoch** = 715 ÷ 8 ≈ **90 steps** — requires `--max-steps -1`

```bash
# ❌ WRONG - Only uses ~240 samples, then stops
modal run unsloth/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen-full --num-epochs 1

# ✅ CORRECT - Uses all 715 samples for 1 epoch
modal run --detach unsloth/modal_coder_base.py --dataset-name lilyzhng/uigen-ui-code-gen-full --num-epochs 1 --max-steps -1
```

### Dataset Preparation

```bash
# Prepare full UIGEN dataset (805 samples) and push to HuggingFace
python finetuning/prepare_full_uigen_dataset.py --push --repo-name lilyzhng/uigen-ui-code-gen-full --yes
```

### Epoch Recommendations

For 715 samples with LoRA:

- **1 epoch** — Default, usually sufficient
- **2 epochs** — If validation loss still improving after 1 epoch
- **3+ epochs** — Higher overfitting risk, use with caution

---

## Best Practices

1. **Always use `--detach` for Modal training jobs**
   - Example: `modal run --detach unsloth/modal_coder_base.py --num-epochs 1 --max-steps -1`
   
2. **Models automatically push to HuggingFace** (default: `push_to_hub=True`)
   - Ensure HF token is configured: `modal secret create hf-secret HF_TOKEN=your_token`
   - No need to manually save/upload models
   
3. **Monitor jobs via Modal web dashboard** instead of local terminal
   - Dashboard: `https://modal.com/apps`
   - Or use: `modal app logs <app-id>`
   
4. **Use LoRA adapters instead of merged models**
   - Faster training completion
   - Easier to version and share
   - Can merge later if needed
   
5. **Set appropriate timeouts** in Modal function decorators
   
6. **Use retries** for transient failures

7. **Modal volumes are temporary storage only**
   - Used during training for checkpoints
   - Not for permanent model hosting
   - Final models live on HuggingFace
