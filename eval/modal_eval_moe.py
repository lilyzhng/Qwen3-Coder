"""
Evaluate Qwen3-Coder-Next-Base (80B MoE) on Modal with GPU.

Compares a base model against a finetuned LoRA model on the same test data.
Uses vLLM for fast inference. Both runs are logged to W&B for easy comparison.

Usage:
    # Evaluate both base and finetuned models (default: 20 samples)
    modal run Qwen3-Coder/eval/modal_eval_moe.py

    # Custom limit
    modal run Qwen3-Coder/eval/modal_eval_moe.py --limit 50

    # Custom LoRA model
    modal run Qwen3-Coder/eval/modal_eval_moe.py --lora-model lilyzhng/my-lora-adapter

    # Only evaluate base model (no LoRA)
    modal run Qwen3-Coder/eval/modal_eval_moe.py --base-only

    # FP8 base + LoRA on H200 (set GPU_CONFIG = 'H200' in file; fits in 141GB)
    modal run Qwen3-Coder/eval/modal_eval_moe.py --fp8-base --lora-model lilyzhng/my-lora-adapter

    # Skip judge (faster, no scoring)
    modal run Qwen3-Coder/eval/modal_eval_moe.py --no-judge
"""

from dataclasses import dataclass
import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App("uiux-eval-moe")

# Container image — uses vLLM for fast inference
eval_image = (
    modal.Image.from_registry('nvidia/cuda:12.8.0-devel-ubuntu22.04', add_python='3.11')
    .apt_install('git', 'build-essential')
    .pip_install(
        'vllm',
        'flashinfer-python',
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

# GPU config — B200 for 80B MoE base eval; for LoRA eval use H200 or H100 (vLLM MoE+LoRA kernel fails on B200)
GPU_CONFIG = "H200"
TIMEOUT_HOURS = 4

# Pre-quantized FP8 base for base-vs-LoRA comparison on H200 (fits in 141GB).
# LoRA trained on bf16 base is compatible. When using --fp8-base with LoRA, set GPU_CONFIG = "H200" above.
FP8_BASE_MODEL = "unsloth/Qwen3-Coder-Next-FP8"


@dataclass
class EvalConfig:
    """Configuration for side-by-side comparison evaluation."""
    base_model: str = "Qwen/Qwen3-Coder-Next-Base"
    lora_model: str = None  # HuggingFace LoRA adapter ID (e.g. lilyzhng/Qwen3-Coder-Next-Base-swift-r8-...)
    hf_dataset: str = "lilyzhng/uigen-ui-code-gen-full"
    output_base_dir: str = "/results"
    limit: int = 20
    judge_model: str = "google/gemini-3-pro-preview"
    wandb_project: str = "uiux-eval"
    use_judge: bool = True
    base_only: bool = False  # Only evaluate base model, skip LoRA
    use_fp8_base: bool = False  # Use FP8-quantized base (e.g. unsloth/Qwen3-Coder-Next-FP8) to fit on H200 with LoRA
    max_new_tokens: int = 8192


@app.function(
    image=eval_image,
    gpu=GPU_CONFIG,
    cpu=8,
    timeout=int(TIMEOUT_HOURS * 3600),
    volumes={
        "/model_cache": model_cache_vol,
        "/results": results_vol,
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

    # Card template for 3-column comparison: Ground Truth | Base Generation | Finetuned Generation
    WANDB_CARD_TEMPLATE = """\
<div style="font-family: system-ui, -apple-system, sans-serif; width: 100%; box-sizing: border-box;">
  <div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
    <div style="font-size: 14px; color: #666; margin-bottom: 4px;">ID: {sample_id} &bull; Base Score: {base_score}/10 &bull; Finetuned Score: {lora_score}/10</div>
    <div style="font-size: 16px; font-weight: 600; color: #1a1a1a;">{prompt}</div>
  </div>
  <div style="display: flex; gap: 16px; margin-bottom: 16px; width: 100%;">
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #27ae60; margin-bottom: 8px;">Ground Truth</div>
      <img src="data:image/png;base64,{gt_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #3498db; margin-bottom: 8px;">Base Model</div>
      <img src="data:image/png;base64,{base_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #9b59b6; margin-bottom: 8px;">Finetuned Model</div>
      <img src="data:image/png;base64,{lora_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
  </div>
  <div style="background: #fafafa; border-radius: 6px; padding: 12px; font-size: 14px;">
    <div style="margin-bottom: 8px;">
      <span style="font-weight: 600; color: #3498db;">Base Failure Modes:</span> {base_failure_modes}<br/>
      <span style="font-weight: 600; color: #3498db;">Base Reasoning:</span> {base_reasoning}
    </div>
    <div>
      <span style="font-weight: 600; color: #9b59b6;">Finetuned Failure Modes:</span> {lora_failure_modes}<br/>
      <span style="font-weight: 600; color: #9b59b6;">Finetuned Reasoning:</span> {lora_reasoning}
    </div>
  </div>
</div>"""

    # 2-column card for base-only mode
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
      <div style="font-size: 14px; font-weight: 600; color: #3498db; margin-bottom: 8px;">Base Model</div>
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

SCORING RUBRIC (start at 10, subtract for issues):
- broken-code (-4): syntax errors, blank page, no output
- broken-layout (-3): elements overlap, misaligned, unusable
- wrong-framework (-2): doesn't use Tailwind CSS
- generic-colors (-2): boring default palette
- no-design-thinking (-2): looks like developer prototype
- missing-states (-1): no hover/transition polish

TASK: {prompt}

GENERATION:
{model_output}

GROUND TRUTH:
{reference}

IMPORTANT: You MUST respond with ONLY a valid JSON object. No other text before or after.
Do not explain your reasoning outside the JSON. Put all reasoning inside the "reasoning" field.

```json
{{"score": <1-10>, "failure_modes": ["<mode1>", "<mode2>"], "reasoning": "<brief explanation with penalty math>"}}
```"""

    # ---------------------------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------------------------
    # Prompt template matching the training format
    PROMPT_TEMPLATE = "# Task: Generate HTML/CSS code using Tailwind CSS\n# Requirements: {requirements}\n\n"

    def load_test_data(hf_dataset: str) -> list[dict]:
        """Load test data from HuggingFace dataset."""
        from datasets import load_dataset

        dataset = load_dataset(hf_dataset, split="test")
        samples = []

        for i, item in enumerate(dataset):
            text = item["text"]
            lines = text.split("\n")
            requirements = ""
            for line in lines:
                if line.startswith("# Requirements:"):
                    requirements = line.replace("# Requirements:", "").strip()
                    break

            full_prompt = PROMPT_TEMPLATE.format(requirements=requirements)

            code_match = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.DOTALL)
            answer = code_match.group(1).strip() if code_match else ""

            samples.append({
                "id": f"test_{i}",
                "question": full_prompt,
                "requirements": requirements,
                "answer": answer,
            })

        return samples

    def extract_code(response_text: str) -> str:
        pattern = r"```(?:html|css|tsx|jsx|vue)?\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            return "\n".join(matches)
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

    def judge_output(client, judge_model, prompt, model_output, reference, gen_img, gt_img, max_retries: int = 2) -> dict:
        judge_text = JUDGE_PROMPT.format(
            prompt=prompt,
            model_output=model_output[:4000],
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

                raw_content = raw_content.strip()

                content = raw_content
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                    content = re.sub(r"\n?```\s*$", "", content)

                json_match = re.search(r"\{[^{}]*(?:\[[^\[\]]*\][^{}]*)*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    json_str = re.sub(r'(?<=: ")(.*?)(?=")', lambda m: m.group(1).replace('\n', ' ').replace('\r', ''), json_str, flags=re.DOTALL)
                    try:
                        parsed = json.loads(json_str)
                        if "score" in parsed:
                            return parsed
                        last_error = "Missing score field"
                        continue
                    except json.JSONDecodeError as e:
                        last_error = f"JSON parse error: {e}"
                        continue

                try:
                    parsed = json.loads(content)
                    if "score" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass

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
        from huggingface_hub import snapshot_download, list_repo_files

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

    # ---------------------------------------------------------------------------
    # Main Execution
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("UIUX Evaluation — Qwen3-Coder-Next-Base (MoE) [vLLM]")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"LoRA model: {config.lora_model or '(none — base only)'}")
    print(f"Use FP8 base: {config.use_fp8_base}")
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
    lora_rank = 64  # default

    if has_lora:
        print(f"\nResolving LoRA adapter: {config.lora_model}")
        lora_local_path = resolve_lora_adapter(config.lora_model)
        print(f"  Adapter path: {lora_local_path}")

        # Read adapter rank from config
        adapter_config_path = os.path.join(lora_local_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            lora_rank = adapter_cfg.get("r", 64)
            print(f"  LoRA rank: {lora_rank}")

    # ---------------------------------------------------------------------------
    # Load Model with vLLM
    # ---------------------------------------------------------------------------
    # vLLM's fused MoE LoRA Triton kernel fails on B200/Blackwell: "tt.elementwise_inline_asm
    # op pipeliner doesn't know how to predicate this op" at gdc_wait(). Fail fast with a clear
    # message instead of crashing during LLM() init.
    if has_lora and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0) or ""
        if "B200" in gpu_name or "Blackwell" in gpu_name:
            raise ValueError(
                "vLLM MoE + LoRA is not supported on B200/Blackwell (fused_moe_lora_op Triton kernel fails). "
                "You are on B200 because GPU_CONFIG at the top of this file is 'B200'. "
                "--fp8-base only selects the FP8 model; it does NOT change the GPU. "
                "Edit modal_eval_moe.py: set GPU_CONFIG = 'H200' (e.g. line ~63), then re-run with "
                "--fp8-base --lora-model <your-adapter> so the job runs on H200."
            )

    # Resolve base model: FP8 pre-quantized for H200 LoRA comparison, or bf16 default
    if config.use_fp8_base:
        model_to_load = FP8_BASE_MODEL
        quantization = "fp8"
    else:
        model_to_load = config.base_model
        quantization = None

    print("\n" + "=" * 60)
    print("LOADING MODEL (vLLM)")
    print("=" * 60)

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model_to_load,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_model_len=16384,
        trust_remote_code=True,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization

    if has_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank

    print(f"Loading model: {model_to_load}" + (" (FP8)" if quantization else ""))
    print(f"  LoRA enabled: {has_lora}")
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
        "use_fp8_base": config.use_fp8_base,
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
    # Batch Generate — base model
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING — BASE MODEL")
    print("=" * 60)

    prompts = [s["question"] for s in samples]

    t0 = time.time()
    base_outputs = llm.generate(prompts, sampling_params)
    base_elapsed = time.time() - t0
    base_texts = [out.outputs[0].text for out in base_outputs]
    total_base_tokens = sum(len(out.outputs[0].token_ids) for out in base_outputs)
    print(f"Base generation done: {len(prompts)} samples, {total_base_tokens} tokens, {base_elapsed:.1f}s ({total_base_tokens/base_elapsed:.0f} tok/s)")

    wandb.log({"base_generation_time_s": round(base_elapsed, 1), "base_total_tokens": total_base_tokens})

    # ---------------------------------------------------------------------------
    # Batch Generate — finetuned model (LoRA)
    # ---------------------------------------------------------------------------
    lora_texts = [""] * len(samples)

    if has_lora:
        print("\n" + "=" * 60)
        print("GENERATING — FINETUNED MODEL (LoRA)")
        print("=" * 60)

        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("finetuned", 1, lora_local_path)

        t0 = time.time()
        lora_outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        lora_elapsed = time.time() - t0
        lora_texts = [out.outputs[0].text for out in lora_outputs]
        total_lora_tokens = sum(len(out.outputs[0].token_ids) for out in lora_outputs)
        print(f"LoRA generation done: {len(prompts)} samples, {total_lora_tokens} tokens, {lora_elapsed:.1f}s ({total_lora_tokens/lora_elapsed:.0f} tok/s)")

        wandb.log({"lora_generation_time_s": round(lora_elapsed, 1), "lora_total_tokens": total_lora_tokens})

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
        gt_html = wrap_in_html(reference, f"GT-{sample_id}")

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
            print("  Judging base model output...")
            base_judgment = judge_output(
                openrouter_client, config.judge_model,
                question, base_extracted, reference, base_img, gt_img
            )
            base_judgments.append(base_judgment)
            print(f"  Base Score: {base_judgment.get('score', '?')}/10")

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
    print(f"Base Model Avg Score: {base_avg:.1f}/10")
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
    base_model: str = "Qwen/Qwen3-Coder-Next-Base",
    lora_model: str = None,
    base_only: bool = False,
    fp8_base: bool = False,
    download_only: bool = False,
    run_name: str = None,
    local_output: str = "wandb/eval_results",
    max_new_tokens: int = 8192,
):
    """Run UIUX evaluation for Qwen3-Coder-Next-Base (MoE) on Modal.

    Side-by-side comparison in W&B:
    - Column 1: Ground Truth
    - Column 2: Base Model Generation
    - Column 3: Finetuned Model Generation (if --lora-model provided)

    Args:
        limit: Number of test samples to evaluate
        no_judge: Skip LLM judging
        base_model: Base model HF ID (ignored if fp8_base=True)
        lora_model: Finetuned LoRA adapter HF ID
        base_only: Only evaluate base model (no LoRA comparison)
        fp8_base: Use FP8-quantized base (unsloth/Qwen3-Coder-Next-FP8) to fit on H200 with LoRA
        download_only: Only download existing results
        run_name: Specific run name to download (used with --download-only)
        local_output: Base directory for local results
        max_new_tokens: Max tokens to generate per sample
    """
    import os
    import time

    if not download_only:
        if not base_only and not lora_model:
            print("WARNING: No --lora-model provided. Running base-only evaluation.")
            print("  To compare base vs finetuned, pass --lora-model <hf_repo_id>")
            base_only = True

        config = EvalConfig(
            base_model=base_model,
            lora_model=lora_model,
            limit=limit,
            use_judge=not no_judge,
            base_only=base_only,
            use_fp8_base=fp8_base,
            max_new_tokens=max_new_tokens,
        )

        print("Starting UIUX evaluation on Modal (vLLM backend)...")
        print(f"  Base model: {config.base_model}" + (" (FP8)" if config.use_fp8_base else ""))
        print(f"  LoRA model: {config.lora_model or '(base only)'}")
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
