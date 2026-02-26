"""
Evaluate Qwen3-Coder-Next (80B MoE, instruct) on Modal with vLLM FP8 inference.

Compares the instruct model against a finetuned LoRA adapter on the same test data.
Uses vLLM for fast inference on H200. Both runs are logged to W&B for easy comparison.

For base-model evaluation, use modal_eval_moe.py instead.

Usage:
    # FP8 instruct + LoRA adapter on H200
    modal run --detach Qwen3-Coder/eval/modal_eval_instruct.py \\
      --lora-model lilyzhng/my-lora-adapter

    # Custom limit
    modal run --detach Qwen3-Coder/eval/modal_eval_instruct.py \\
      --lora-model lilyzhng/my-lora-adapter --limit 50

    # Instruct model only (no LoRA comparison)
    modal run --detach Qwen3-Coder/eval/modal_eval_instruct.py --base-only

    # Skip judge (faster, no scoring)
    modal run --detach Qwen3-Coder/eval/modal_eval_instruct.py \\
      --lora-model lilyzhng/my-lora-adapter --no-judge
"""

from dataclasses import dataclass
import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App("uiux-eval-instruct")

# Container image — uses vLLM for fast inference
eval_image = (
    modal.Image.from_registry('nvidia/cuda:12.8.0-devel-ubuntu22.04', add_python='3.11')
    .apt_install('git', 'build-essential')
    .pip_install(
        'vllm==0.15.1',
        'flashinfer-python',
        'peft',
        'datasets',
        'hf-transfer',
        'wandb',
        'openai',
        'python-dotenv',
        'playwright',
    )
    .run_commands('playwright install --with-deps chromium')
    .env({
        'HF_HOME': '/model_cache',
        'HF_HUB_ENABLE_HF_TRANSFER': '1',
        'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
        'VLLM_ATTENTION_BACKEND': 'FLASHINFER',
    })
)

# Persistent volumes
model_cache_vol = modal.Volume.from_name("uiux-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("uiux-eval-results", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# GPU config — H200 for FP8 instruct + LoRA eval
GPU_CONFIG = "H200"
TIMEOUT_HOURS = 4

# FP8-quantized instruct model (Qwen3-Coder-Next, NOT the base model).
# LoRA adapters trained on bf16 instruct base are compatible with FP8 inference.
FP8_MODEL = "unsloth/Qwen3-Coder-Next-FP8"


@dataclass
class EvalConfig:
    """Configuration for instruct model evaluation."""
    base_model: str = "Qwen/Qwen3-Coder-Next"
    lora_model: str = None  # HuggingFace LoRA adapter ID (e.g. lilyzhng/Qwen3-Coder-Next-sft-r8-...)
    hf_dataset: str = "lilyzhng/UIGEN-T1.1-split"
    output_base_dir: str = "/results"
    limit: int = 20
    judge_model: str = "google/gemini-3-pro-preview"
    wandb_project: str = "uiux-eval"
    use_judge: bool = True
    base_only: bool = False  # Only evaluate instruct model, skip LoRA
    use_fp8: bool = True  # FP8 inference (default on — instruct model fits on H200 in FP8)
    max_new_tokens: int = 2048


@app.function(
    image=eval_image,
    gpu=GPU_CONFIG,
    cpu=8,
    timeout=int(TIMEOUT_HOURS * 3600),
    volumes={
        "/model_cache": model_cache_vol,
        "/results": results_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
        modal.Secret.from_name("openrouter-secret"),
    ],
)
def run_evaluation(config: EvalConfig):
    """Run evaluation on both base and finetuned models."""
    import os
    import json
    import time
    import base64
    import re

    import torch
    import wandb
    from openai import OpenAI
    from playwright.sync_api import sync_playwright

    # Print GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_gb = round(gpu.total_memory / 1024**3, 1)
        print(f'GPU: {gpu.name}, {total_gb} GB VRAM')

    # ---------------------------------------------------------------------------
    # Constants and Templates
    # ---------------------------------------------------------------------------
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <title>{title}</title>
</head>
<body>
{content}
</body>
</html>
"""

    # Card template for 3-column comparison: Ground Truth | Instruct Generation | Finetuned Generation
    WANDB_CARD_TEMPLATE = """\
<div style="font-family: system-ui, -apple-system, sans-serif; width: 100%; box-sizing: border-box;">
  <div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
    <div style="font-size: 14px; color: #666; margin-bottom: 4px;">ID: {sample_id} &bull; Instruct Score: {base_score}/10 &bull; Finetuned Score: {lora_score}/10</div>
    <div style="font-size: 16px; font-weight: 600; color: #1a1a1a;">{prompt}</div>
  </div>
  <div style="display: flex; gap: 16px; margin-bottom: 16px; width: 100%;">
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #27ae60; margin-bottom: 8px;">Ground Truth</div>
      <img src="data:image/png;base64,{gt_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #3498db; margin-bottom: 8px;">Instruct Model</div>
      <img src="data:image/png;base64,{base_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #9b59b6; margin-bottom: 8px;">Finetuned Model</div>
      <img src="data:image/png;base64,{lora_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
  </div>
  <div style="background: #fafafa; border-radius: 6px; padding: 12px; font-size: 14px;">
    <div style="margin-bottom: 8px;">
      <span style="font-weight: 600; color: #3498db;">Instruct Failure Modes:</span> {base_failure_modes}<br/>
      <span style="font-weight: 600; color: #3498db;">Instruct Reasoning:</span> {base_reasoning}
    </div>
    <div>
      <span style="font-weight: 600; color: #9b59b6;">Finetuned Failure Modes:</span> {lora_failure_modes}<br/>
      <span style="font-weight: 600; color: #9b59b6;">Finetuned Reasoning:</span> {lora_reasoning}
    </div>
  </div>
</div>"""

    # 2-column card for instruct-only mode (no LoRA)
    WANDB_CARD_BASE_ONLY_TEMPLATE = """\
<div style="font-family: system-ui, -apple-system, sans-serif; width: 100%; box-sizing: border-box;">
  <div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
    <div style="font-size: 14px; color: #666; margin-bottom: 4px;">ID: {sample_id} &bull; Score: {base_score}/10</div>
    <div style="font-size: 16px; font-weight: 600; color: #1a1a1a;">{prompt}</div>
  </div>
  <div style="display: flex; gap: 16px; margin-bottom: 16px; width: 100%;">
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #27ae60; margin-bottom: 8px;">Ground Truth</div>
      <img src="data:image/png;base64,{gt_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #3498db; margin-bottom: 8px;">Instruct Model</div>
      <img src="data:image/png;base64,{base_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
  </div>
  <div style="background: #fafafa; border-radius: 6px; padding: 12px; font-size: 14px;">
    <span style="font-weight: 600; color: #3498db;">Failure Modes:</span> {base_failure_modes}<br/>
    <span style="font-weight: 600; color: #3498db;">Reasoning:</span> {base_reasoning}
  </div>
</div>"""

    JUDGE_PROMPT = """\
You are a UI code quality judge. Rate the generation from 1-10.

CRITICAL: Screenshots are attached below the code. The SCREENSHOT is the ground truth for what the code actually renders. If the screenshot shows a working UI, the code is NOT broken — even if the code snippet appears truncated. Always trust the screenshot over the code text.

SCORING RUBRIC (start at 10, subtract for issues):
- broken-code (-4): screenshot shows blank page or error — NOT just truncated code
- broken-layout (-3): screenshot shows elements overlap, misaligned, unusable
- wrong-framework (-2): doesn't use Tailwind CSS (check for tailwind classes in code)
- generic-colors (-2): boring default palette
- no-design-thinking (-2): looks like developer prototype
- visible-artifacts (-3): screenshot shows raw code, markdown fences, or explanatory text rendered visibly on the page
- missing-states (-1): no hover/transition polish

TASK: {prompt}

GENERATION (may be truncated — refer to the screenshot for actual rendered output):
{model_output}

GROUND TRUTH:
{reference}

Respond with ONLY a JSON object. No other text before or after.

{{"score": <1-10>, "failure_modes": ["<mode1>", "<mode2>"], "reasoning": "<brief explanation>"}}"""

    # ---------------------------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------------------------
    def load_test_data(hf_dataset: str) -> list[dict]:
        """Load test split from HuggingFace dataset.

        Expects lilyzhng/UIGEN-T1.1-split (test split) with columns:
          id, question, answer
        These are loaded directly — no text parsing or regex needed.
        """
        from datasets import load_dataset

        dataset = load_dataset(hf_dataset, split="test")
        samples = []
        for item in dataset:
            samples.append({
                "id": str(item["id"]),
                "question": item["question"],
                "answer": item["answer"],
            })
        return samples

    def extract_code(response_text: str) -> str:
        # Try matching complete fenced code blocks first
        pattern = r"```(?:html|css|tsx|jsx|vue)?\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            return "\n".join(matches)
        # Handle truncated output: opening fence but no closing fence
        # (common when model hits max_new_tokens before closing ```)
        open_pattern = r"```(?:html|css|tsx|jsx|vue)?\s*\n(.*)"
        open_match = re.search(open_pattern, response_text, re.DOTALL)
        if open_match:
            return open_match.group(1).strip()
        stripped = response_text.strip()
        if stripped.startswith("<") or stripped.startswith("<!"):
            return stripped
        return stripped

    def wrap_in_html(code: str, title: str = "UI Output") -> str:
        if "<!DOCTYPE" in code.upper() or "<html" in code.lower():
            if "tailwindcss" not in code:
                code = code.replace(
                    "<head>",
                    '<head>\n  <script src="https://cdn.tailwindcss.com"></script>',
                    1,
                )
            return code
        return HTML_TEMPLATE.format(title=title, content=code)

    def render_screenshot(html_path: str, screenshot_path: str, browser) -> bool:
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"file://{os.path.abspath(html_path)}", wait_until="networkidle")
            page.wait_for_timeout(1000)
            page.screenshot(path=screenshot_path, full_page=True)
            page.close()
            return True
        except Exception as e:
            print(f"  Screenshot failed: {e}")
            return False

    def image_to_base64(path: str) -> str:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return ""

    def extract_json_from_response(raw_content: str) -> dict | None:
        """Extract a JSON object from a judge response using brace-counting.

        Handles: clean JSON, ```json fences, text preamble before JSON,
        escaped quotes inside strings, and nested structures.
        Returns the parsed dict or None if no valid JSON found.
        """
        if not raw_content or not raw_content.strip():
            return None

        text = raw_content.strip()

        # Strip markdown fences if present (anywhere in the text)
        text = re.sub(r'```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*', '', text)

        # Find the first '{' and extract JSON by brace counting
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        end = start

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if ch == '\\' and in_string:
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        # Sanitize newlines inside string values (common in judge reasoning)
        def sanitize_string_values(s):
            result = []
            in_str = False
            esc = False
            for c in s:
                if esc:
                    result.append(c)
                    esc = False
                    continue
                if c == '\\' and in_str:
                    result.append(c)
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    result.append(c)
                    continue
                if in_str and c in ('\n', '\r'):
                    result.append(' ')
                    continue
                result.append(c)
            return ''.join(result)

        def try_parse(json_str):
            json_str = sanitize_string_values(json_str)
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and 'score' in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
            return None

        if depth == 0:
            json_str = text[start:end]
            return try_parse(json_str)

        # Truncated JSON — judge response was cut off.
        # Try to repair by closing open strings/arrays/objects.
        json_str = text[start:]

        # Attempt 1: close the string + close braces/brackets
        repair = json_str.rstrip().rstrip(',')
        if in_string:
            repair += '"'
        bracket_depth = 0
        str_mode = False
        esc_mode = False
        for c in repair:
            if esc_mode:
                esc_mode = False
                continue
            if c == '\\' and str_mode:
                esc_mode = True
                continue
            if c == '"':
                str_mode = not str_mode
                continue
            if str_mode:
                continue
            if c == '[':
                bracket_depth += 1
            elif c == ']':
                bracket_depth -= 1
        repair += ']' * max(bracket_depth, 0)
        repair += '}' * depth

        result = try_parse(repair)
        if result:
            return result

        # Attempt 2: truncate to the last complete key-value pair
        last_comma = -1
        str_mode = False
        esc_mode = False
        for idx, c in enumerate(json_str):
            if esc_mode:
                esc_mode = False
                continue
            if c == '\\' and str_mode:
                esc_mode = True
                continue
            if c == '"':
                str_mode = not str_mode
                continue
            if str_mode:
                continue
            if c == ',':
                last_comma = idx

        if last_comma > 0:
            truncated = json_str[:last_comma]
            bracket_depth = 0
            str_mode = False
            esc_mode = False
            for c in truncated:
                if esc_mode:
                    esc_mode = False
                    continue
                if c == '\\' and str_mode:
                    esc_mode = True
                    continue
                if c == '"':
                    str_mode = not str_mode
                    continue
                if str_mode:
                    continue
                if c == '[':
                    bracket_depth += 1
                elif c == ']':
                    bracket_depth -= 1
            truncated += ']' * max(bracket_depth, 0)
            truncated += '}'
            result = try_parse(truncated)
            if result:
                return result

        return None

    def judge_output(client, judge_model, prompt, model_output, reference, gen_img, gt_img, max_retries: int = 2) -> dict:
        judge_text = JUDGE_PROMPT.format(
            prompt=prompt,
            model_output=model_output[:8000],
            reference=reference[:4000],
        )
        content_parts = [{"type": "text", "text": judge_text}]

        gen_exists = gen_img and os.path.exists(gen_img)
        gt_exists = gt_img and os.path.exists(gt_img)

        if gen_exists:
            gen_b64 = image_to_base64(gen_img)
            if gen_b64:
                content_parts.append({"type": "text", "text": "\n\nGeneration screenshot:"})
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{gen_b64}"},
                })
        if gt_exists:
            gt_b64 = image_to_base64(gt_img)
            if gt_b64:
                content_parts.append({"type": "text", "text": "\n\nGround truth screenshot:"})
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{gt_b64}"},
                })

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"    Retry {attempt}/{max_retries}...")
                    time.sleep(1)

                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": content_parts}],
                    max_tokens=2048,
                    temperature=0.0,
                )
                raw_content = response.choices[0].message.content
                if raw_content is None or len(raw_content.strip()) == 0:
                    last_error = "Empty response"
                    continue

                parsed = extract_json_from_response(raw_content)
                if parsed:
                    return parsed

                last_error = f"No JSON in response: {raw_content[:100]}..."
                continue

            except Exception as e:
                last_error = f"API error: {e}"
                continue

        return {"score": 0, "failure_modes": ["judge-error"], "reasoning": f"After {max_retries+1} attempts: {last_error}"}

    def resolve_lora_adapter(lora_repo_id: str) -> str:
        """Download LoRA adapter from HF and return local path.

        ms-swift saves adapters in subdirectories like v0-.../checkpoint-N/.
        This function auto-detects the adapter location.
        """
        from huggingface_hub import snapshot_download

        # Download the full repo
        repo_path = snapshot_download(lora_repo_id)

        # Check if adapter_config.json is at root
        if os.path.exists(os.path.join(repo_path, "adapter_config.json")):
            return repo_path

        # Scan for adapter_config.json in subdirectories
        for root, dirs, files in os.walk(repo_path):
            if "adapter_config.json" in files:
                print(f"  Found adapter at subdirectory: {os.path.relpath(root, repo_path)}")
                return root

        raise ValueError(f"No adapter_config.json found in {lora_repo_id}")

    MOE_EXPERT_MODULES = {'gate_up_proj', 'down_proj', 'gate_proj', 'up_proj'}

    def adapter_needs_merge(adapter_path: str) -> bool:
        """Check if a LoRA adapter targets MoE expert FFN layers.

        vLLM's runtime LoRA works for attention modules but crashes on MoE
        expert layers (pack_moe AssertionError). Returns True if the adapter
        has any expert-layer targets, meaning we must merge-then-load.
        """
        config_path = os.path.join(adapter_path, 'adapter_config.json')
        with open(config_path) as f:
            adapter_cfg = json.load(f)
        target_modules = set(adapter_cfg.get('target_modules', []))
        has_expert_modules = bool(target_modules & MOE_EXPERT_MODULES)
        lora_rank = adapter_cfg.get('r', 8)
        print(f'  Target modules: {sorted(target_modules)}')
        print(f'  Rank: {lora_rank}')
        print(f'  Has MoE expert layers: {has_expert_modules}')
        if has_expert_modules:
            print('  → Will use merge-then-load (vLLM pack_moe incompatible)')
        else:
            print('  → Will use vLLM runtime LoRA (fast path)')
        return has_expert_modules, lora_rank

    # ---------------------------------------------------------------------------
    # Main Execution
    # ---------------------------------------------------------------------------
    actual_model = FP8_MODEL if config.use_fp8 else config.base_model

    print("=" * 60)
    print("UIUX Evaluation — Qwen3-Coder-Next (instruct, MoE) [vLLM]")
    print("=" * 60)
    print(f"Model: {actual_model}" + (" (FP8)" if config.use_fp8 else ""))
    print(f"LoRA model: {config.lora_model or '(none — instruct only)'}")
    print(f"FP8 inference: {config.use_fp8}")
    print(f"HF Dataset: {config.hf_dataset}")
    print(f"Limit: {config.limit}")
    print(f"Use judge: {config.use_judge}")
    print(f"Base only: {config.base_only}")
    print()

    # Load test data from HuggingFace
    samples = load_test_data(config.hf_dataset)
    if config.limit:
        samples = samples[:config.limit]
    print(f"Loaded {len(samples)} test samples")

    # Setup OpenRouter client for judging
    openrouter_client = None
    if config.use_judge:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            openrouter_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key)
            print(f"Judge: {config.judge_model}")
        else:
            print("WARNING: No OPENROUTER_API_KEY found, disabling judge")
            config.use_judge = False

    # ---------------------------------------------------------------------------
    # Resolve LoRA adapter (download + find subdirectory)
    # ---------------------------------------------------------------------------
    has_lora = not config.base_only and config.lora_model is not None
    lora_local_path = None
    use_merge_path = False
    adapter_rank = 8

    if has_lora:
        print(f"\nResolving LoRA adapter: {config.lora_model}")
        lora_local_path = resolve_lora_adapter(config.lora_model)
        print(f"  Adapter path: {lora_local_path}")
        use_merge_path, adapter_rank = adapter_needs_merge(lora_local_path)

    # ---------------------------------------------------------------------------
    # Load Model with vLLM
    # ---------------------------------------------------------------------------
    # Two LoRA eval strategies depending on adapter target modules:
    #   1. Attention-only adapters → vLLM runtime LoRA (fast, no reload needed)
    #   2. MoE expert-layer adapters → merge-then-load (vLLM pack_moe crashes)

    # Resolve model: FP8 (default) or bf16
    if config.use_fp8:
        model_to_load = FP8_MODEL
        quantization = "fp8"
    else:
        model_to_load = config.base_model
        quantization = None

    print("\n" + "=" * 60)
    print("LOADING MODEL (vLLM)")
    print("=" * 60)

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    use_runtime_lora = has_lora and not use_merge_path

    llm_kwargs = dict(
        model=model_to_load,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        trust_remote_code=True,
        compilation_config=CompilationConfig(cudagraph_mode="PIECEWISE"),
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    if use_runtime_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = adapter_rank

    print(f"Loading model: {model_to_load}" + (" (FP8)" if quantization else ""))
    if use_runtime_lora:
        print(f"  Runtime LoRA enabled (rank={adapter_rank})")
    llm = LLM(**llm_kwargs)
    print(f"Model loaded! Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        max_tokens=config.max_new_tokens,
    )

    # ---------------------------------------------------------------------------
    # Initialize W&B early (before generation so we can track progress)
    # ---------------------------------------------------------------------------
    base_short = model_to_load.split("/")[-1]
    if has_lora:
        lora_short = config.lora_model.split("/")[-1]
        run_name = f"comparison-{base_short}-vs-{lora_short}-{time.strftime('%m%d-%H%M')}"
    else:
        run_name = f"base-{base_short}-{time.strftime('%m%d-%H%M')}"

    output_dir = os.path.join(config.output_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    screenshots_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    wandb_config = {
        "base_model": model_to_load,
        "use_fp8": config.use_fp8,
        "lora_model": config.lora_model,
        "judge_model": config.judge_model if config.use_judge else "none",
        "num_samples": len(samples),
        "base_only": config.base_only,
        "inference_backend": "vllm",
    }

    run = wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=wandb_config,
    )
    print(f"W&B run: {run.url}")

    # ---------------------------------------------------------------------------
    # Batch Generate — instruct model
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING — INSTRUCT MODEL")
    print("=" * 60)

    # Build chat-format conversations so the model's ChatML template is applied.
    # llm.chat() wraps each question in <|im_start|>user...<|im_end|> automatically,
    # matching the format the instruct model and LoRA adapters were trained with.
    conversations = [[{"role": "user", "content": s["question"]}] for s in samples]

    t0 = time.time()
    base_outputs = llm.chat(conversations, sampling_params)
    base_elapsed = time.time() - t0
    base_texts = [out.outputs[0].text for out in base_outputs]
    total_base_tokens = sum(len(out.outputs[0].token_ids) for out in base_outputs)
    print(f"Instruct generation done: {len(conversations)} samples, {total_base_tokens} tokens, {base_elapsed:.1f}s ({total_base_tokens/base_elapsed:.0f} tok/s)")

    vllm_cache_vol.commit()

    wandb.log({"instruct_generation_time_s": round(base_elapsed, 1), "instruct_total_tokens": total_base_tokens})

    # ---------------------------------------------------------------------------
    # Batch Generate — finetuned model (LoRA)
    # ---------------------------------------------------------------------------
    lora_texts = [""] * len(samples)

    if has_lora and use_runtime_lora:
        # Fast path: attention-only adapter → vLLM runtime LoRA (no reload)
        from vllm.lora.request import LoRARequest

        print("\n" + "=" * 60)
        print("GENERATING — FINETUNED MODEL (vLLM runtime LoRA)")
        print("=" * 60)
        print(f"  Adapter: {lora_local_path} (rank={adapter_rank})")

        lora_request = LoRARequest("finetuned", 1, lora_local_path)

        t0 = time.time()
        lora_outputs = llm.chat(conversations, sampling_params, lora_request=lora_request)
        lora_elapsed = time.time() - t0
        lora_texts = [out.outputs[0].text for out in lora_outputs]
        total_lora_tokens = sum(len(out.outputs[0].token_ids) for out in lora_outputs)
        print(f"LoRA generation done: {len(conversations)} samples, {total_lora_tokens} tokens, {lora_elapsed:.1f}s ({total_lora_tokens/lora_elapsed:.0f} tok/s)")

        wandb.log({"lora_generation_time_s": round(lora_elapsed, 1), "lora_total_tokens": total_lora_tokens})

        del llm
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    elif has_lora and use_merge_path:
        # Slow path: MoE expert-layer adapter → merge weights then reload.
        # vLLM's pack_moe crashes with AssertionError on gate_up_proj/down_proj
        # LoRA weights, so we merge on CPU and reload the full model.
        print("\n" + "=" * 60)
        print("GENERATING — FINETUNED MODEL (LoRA merge-then-load)")
        print("=" * 60)

        print("Destroying vLLM instance to free GPU memory...")
        del llm
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print("Merging LoRA adapter into base model...")
        print(f"  Base: {config.base_model}")
        print(f"  Adapter: {lora_local_path}")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        merge_dir = "/tmp/merged_model"

        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        print("  Loading base model for merge (this takes a few minutes)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("  Applying LoRA adapter...")
        merged_model = PeftModel.from_pretrained(base_model, lora_local_path)
        print("  Merging weights...")
        merged_model = merged_model.merge_and_unload()
        print(f"  Saving merged model to {merge_dir}...")
        merged_model.save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)

        del base_model, merged_model
        torch.cuda.empty_cache()
        gc.collect()
        print("  Merge complete. Loading merged model into vLLM...")

        merged_llm_kwargs = dict(
            model=merge_dir,
            dtype="bfloat16",
            gpu_memory_utilization=0.92,
            max_model_len=4096,
            trust_remote_code=True,
            compilation_config=CompilationConfig(cudagraph_mode="PIECEWISE"),
        )
        if config.use_fp8:
            merged_llm_kwargs["quantization"] = "fp8"

        llm_merged = LLM(**merged_llm_kwargs)
        print(f"  Merged model loaded! Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

        t0 = time.time()
        lora_outputs = llm_merged.chat(conversations, sampling_params)
        lora_elapsed = time.time() - t0
        lora_texts = [out.outputs[0].text for out in lora_outputs]
        total_lora_tokens = sum(len(out.outputs[0].token_ids) for out in lora_outputs)
        print(f"LoRA generation done: {len(conversations)} samples, {total_lora_tokens} tokens, {lora_elapsed:.1f}s ({total_lora_tokens/lora_elapsed:.0f} tok/s)")

        wandb.log({"lora_generation_time_s": round(lora_elapsed, 1), "lora_total_tokens": total_lora_tokens})

        del llm_merged
        torch.cuda.empty_cache()
        gc.collect()
    else:
        del llm
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # ---------------------------------------------------------------------------
    # Process Results (screenshots, judging, W&B)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROCESSING RESULTS")
    print("=" * 60)

    # Launch browser
    print("Launching browser...")
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)

    base_judgments = []
    lora_judgments = []

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        question = sample["question"]
        requirements = sample.get("requirements", question[:100])
        reference = sample["answer"]

        base_raw = base_texts[i]
        lora_raw = lora_texts[i]

        print(f"\n[{i+1}/{len(samples)}] ID={sample_id}: {requirements[:50]}...")

        # Process HTML
        base_extracted = extract_code(base_raw)
        base_html = wrap_in_html(base_extracted, f"Base-{sample_id}")
        gt_extracted = extract_code(reference)
        gt_html = wrap_in_html(gt_extracted, f"GT-{sample_id}")

        # Save files
        base_path = os.path.join(output_dir, f"{sample_id}_base.html")
        gt_path = os.path.join(output_dir, f"{sample_id}_gt.html")
        base_img = os.path.join(screenshots_dir, f"{sample_id}_base.png")
        gt_img = os.path.join(screenshots_dir, f"{sample_id}_gt.png")

        with open(base_path, "w") as f:
            f.write(base_html)
        with open(gt_path, "w") as f:
            f.write(gt_html)
        with open(os.path.join(output_dir, f"{sample_id}_base_raw.txt"), "w") as f:
            f.write(base_raw)

        if has_lora:
            lora_extracted = extract_code(lora_raw)
            lora_html = wrap_in_html(lora_extracted, f"Finetuned-{sample_id}")
            lora_path = os.path.join(output_dir, f"{sample_id}_lora.html")
            lora_img = os.path.join(screenshots_dir, f"{sample_id}_lora.png")
            with open(lora_path, "w") as f:
                f.write(lora_html)
            with open(os.path.join(output_dir, f"{sample_id}_lora_raw.txt"), "w") as f:
                f.write(lora_raw)

        # Screenshots
        print("  Rendering screenshots...")
        render_screenshot(base_path, base_img, browser)
        render_screenshot(gt_path, gt_img, browser)
        if has_lora:
            render_screenshot(lora_path, lora_img, browser)

        # Judge
        base_judgment = {}
        lora_judgment = {}

        if config.use_judge and openrouter_client:
            print("  Judging instruct model output...")
            base_judgment = judge_output(
                openrouter_client, config.judge_model,
                question, base_extracted, reference, base_img, gt_img
            )
            base_judgments.append(base_judgment)
            print(f"  Instruct Score: {base_judgment.get('score', '?')}/10")

            if has_lora:
                print("  Judging finetuned model output...")
                lora_judgment = judge_output(
                    openrouter_client, config.judge_model,
                    question, lora_extracted, reference, lora_img, gt_img
                )
                lora_judgments.append(lora_judgment)
                print(f"  Finetuned Score: {lora_judgment.get('score', '?')}/10")

            # Log to W&B
            if has_lora:
                card_html = WANDB_CARD_TEMPLATE.format(
                    sample_id=sample_id,
                    prompt=requirements[:200],
                    base_score=base_judgment.get("score", "—"),
                    lora_score=lora_judgment.get("score", "—"),
                    gt_b64=image_to_base64(gt_img),
                    base_b64=image_to_base64(base_img),
                    lora_b64=image_to_base64(lora_img),
                    base_failure_modes=", ".join(base_judgment.get("failure_modes", [])) or "—",
                    base_reasoning=base_judgment.get("reasoning", "—"),
                    lora_failure_modes=", ".join(lora_judgment.get("failure_modes", [])) or "—",
                    lora_reasoning=lora_judgment.get("reasoning", "—"),
                )
                wandb.log({
                    f"samples/{sample_id}": wandb.Html(card_html),
                    "base_score": base_judgment.get("score", 0),
                    "lora_score": lora_judgment.get("score", 0),
                    "score_diff": lora_judgment.get("score", 0) - base_judgment.get("score", 0),
                    "sample_idx": i,
                })
            else:
                card_html = WANDB_CARD_BASE_ONLY_TEMPLATE.format(
                    sample_id=sample_id,
                    prompt=requirements[:200],
                    base_score=base_judgment.get("score", "—"),
                    gt_b64=image_to_base64(gt_img),
                    base_b64=image_to_base64(base_img),
                    base_failure_modes=", ".join(base_judgment.get("failure_modes", [])) or "—",
                    base_reasoning=base_judgment.get("reasoning", "—"),
                )
                wandb.log({
                    f"samples/{sample_id}": wandb.Html(card_html),
                    "base_score": base_judgment.get("score", 0),
                    "sample_idx": i,
                })

            # Save judgments
            with open(os.path.join(output_dir, f"{sample_id}_base_judgment.json"), "w") as f:
                json.dump(base_judgment, f, indent=2)
            if has_lora:
                with open(os.path.join(output_dir, f"{sample_id}_lora_judgment.json"), "w") as f:
                    json.dump(lora_judgment, f, indent=2)

    # Cleanup
    browser.close()
    pw.stop()

    # Summary
    base_avg = 0
    lora_avg = 0
    if base_judgments:
        base_scores = [j.get("score", 0) for j in base_judgments if j.get("score", 0) > 0]
        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0
        wandb.summary["base_avg_score"] = round(base_avg, 2)

    if lora_judgments:
        lora_scores = [j.get("score", 0) for j in lora_judgments if j.get("score", 0) > 0]
        lora_avg = sum(lora_scores) / len(lora_scores) if lora_scores else 0
        wandb.summary["lora_avg_score"] = round(lora_avg, 2)

    if has_lora:
        wandb.summary["score_improvement"] = round(lora_avg - base_avg, 2)

    wandb.summary["num_samples"] = len(samples)
    wandb.summary["inference_backend"] = "vllm"

    print(f"\n{'='*40}")
    print(f"Instruct Model Avg Score: {base_avg:.1f}/10")
    if has_lora:
        print(f"Finetuned Model Avg Score: {lora_avg:.1f}/10")
        print(f"Improvement: {lora_avg - base_avg:+.1f}")
    print(f"{'='*40}")

    wandb.finish()
    return {
        "base_avg_score": base_avg if base_judgments else None,
        "lora_avg_score": lora_avg if lora_judgments else None,
        "run_url": run.url,
        "run_name": run_name,
    }


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": results_vol},
    timeout=600,
)
def list_results(run_name: str = None):
    """List files in the results volume."""
    import os

    base_path = f"/results/{run_name}" if run_name else "/results"
    if not os.path.exists(base_path):
        print(f"Path does not exist: {base_path}")
        return []

    files = []
    for root, dirs, filenames in os.walk(base_path):
        for f in filenames:
            files.append(os.path.join(root, f))
    return files


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": results_vol},
    timeout=600,
)
def read_file(path: str) -> bytes:
    """Read a file from the results volume."""
    with open(path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    limit: int = 20,
    no_judge: bool = False,
    lora_model: str = None,
    base_only: bool = False,
    no_fp8: bool = False,
    download_only: bool = False,
    run_name: str = None,
    local_output: str = "wandb/eval_results",
    max_new_tokens: int = 2048,
):
    """Run UIUX evaluation for Qwen3-Coder-Next (instruct, MoE) on Modal.

    Uses FP8 inference by default (unsloth/Qwen3-Coder-Next-FP8 on H200).

    Side-by-side comparison in W&B:
    - Column 1: Ground Truth
    - Column 2: Instruct Model Generation
    - Column 3: Finetuned Model Generation (if --lora-model provided)

    Args:
        limit: Number of test samples to evaluate
        no_judge: Skip LLM judging
        lora_model: Finetuned LoRA adapter HF ID
        base_only: Only evaluate instruct model (no LoRA comparison)
        no_fp8: Disable FP8 quantization (use bf16 instead — requires more VRAM)
        download_only: Only download existing results
        run_name: Specific run name to download (used with --download-only)
        local_output: Base directory for local results
        max_new_tokens: Max tokens to generate per sample
    """
    import os
    import time

    if not download_only:
        if not base_only and not lora_model:
            print("WARNING: No --lora-model provided. Running instruct-only evaluation.")
            print("  To compare instruct vs finetuned, pass --lora-model <hf_repo_id>")
            base_only = True

        use_fp8 = not no_fp8
        actual_model = FP8_MODEL if use_fp8 else "Qwen/Qwen3-Coder-Next"

        config = EvalConfig(
            lora_model=lora_model,
            limit=limit,
            use_judge=not no_judge,
            base_only=base_only,
            use_fp8=use_fp8,
            max_new_tokens=max_new_tokens,
        )

        print("Starting UIUX evaluation on Modal (vLLM backend)...")
        print(f"  Model: {actual_model}" + (" (FP8)" if use_fp8 else ""))
        print(f"  LoRA model: {config.lora_model or '(instruct only)'}")
        print(f"  Limit: {config.limit}")
        print(f"  Max new tokens: {config.max_new_tokens}")
        print(f"  Use judge: {config.use_judge}")

        results = run_evaluation.remote(config)
        print("\nResults:", results)
        run_name = results.get("run_name")
    elif not run_name:
        # List available runs instead of downloading everything
        print("Available runs on the results volume:")
        all_files = list_results.remote(run_name=None)
        runs = sorted({f.split("/results/")[1].split("/")[0] for f in all_files if "/results/" in f})
        for r in runs:
            print(f"  {r}")
        print("\nTo download a specific run: modal run ... --download-only --run-name <name>")
        return

    # Download results
    subfolder = run_name or f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = os.path.join(local_output, subfolder)

    print(f"\nDownloading results to {output_dir}/...")
    files = list_results.remote(run_name=run_name)
    print(f"Found {len(files)} files")

    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0
    for remote_path in files:
        if run_name:
            relative_path = remote_path.replace(f"/results/{run_name}/", "")
        else:
            relative_path = remote_path.replace("/results/", "")
        local_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            content = read_file.remote(remote_path)
            with open(local_path, "wb") as f:
                f.write(content)
            downloaded += 1
        except Exception as e:
            print(f"  Failed: {relative_path} - {e}")

    print(f"\nDownloaded {downloaded}/{len(files)} files to {output_dir}/")
