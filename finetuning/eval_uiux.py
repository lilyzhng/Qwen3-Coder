"""
Evaluate a model's UI/UX code generation quality using the UIGEN test split.

Supports three modes:
1. OpenRouter API: Send prompts to hosted models via OpenRouter
2. Local LoRA: Load a finetuned LoRA model locally (for Modal/GPU servers)
3. Local Base: Load a base model locally for comparison

Uses a judge model (Gemini 3 Pro via OpenRouter) to automatically score and
tag failure modes. Also saves generated HTML with Tailwind CDN and screenshots.

Usage:
    # Evaluate via OpenRouter API
    python finetuning/eval_uiux.py \
        --model "qwen/qwen3-coder-next" \
        --test-data data/uigen_test.jsonl \
        --output-dir eval_results/baseline \
        --limit 5

    # Evaluate a local LoRA model (for Modal)
    python finetuning/eval_uiux.py \
        --lora-model "lilyzhng/Qwen2.5-Coder-14B-r32-20260208-164425" \
        --test-data data/uigen_test.jsonl \
        --output-dir eval_results/lora_eval \
        --limit 10

    # Evaluate a local base model (for comparison)
    python finetuning/eval_uiux.py \
        --local-model "Qwen/Qwen2.5-Coder-14B-Instruct" \
        --test-data data/uigen_test.jsonl \
        --output-dir eval_results/base_eval \
        --limit 10

    # Without judge (manual review only)
    python finetuning/eval_uiux.py \
        --lora-model "lilyzhng/Qwen2.5-Coder-14B-r32-20260208-164425" \
        --test-data data/uigen_test.jsonl \
        --output-dir eval_results/lora_eval \
        --no-judge

Comparison workflow:
    # 1. Evaluate base model
    python finetuning/eval_uiux.py \
        --local-model "Qwen/Qwen2.5-Coder-14B-Instruct" \
        --test-data data/uigen_test.jsonl \
        --output-dir eval_results/base \
        --limit 20

    # 2. Evaluate LoRA model
    python finetuning/eval_uiux.py \
        --lora-model "lilyzhng/Qwen2.5-Coder-14B-r32-20260208-164425" \
        --test-data data/uigen_test.jsonl \
        --output-dir eval_results/lora \
        --limit 20

    # 3. Compare in W&B dashboard - both runs are in the same project
"""

import argparse
import base64
import json
import os
import re
import sys
import time

import wandb
from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright

load_dotenv(override=True)

# Ensure print output is flushed immediately (important for long-running scripts)
sys.stdout.reconfigure(line_buffering=True)

# Defaults
DEFAULT_JUDGE_MODEL = "google/gemini-3-pro-preview"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
WANDB_PROJECT = "uiux-eval"

# Global variables for local model (initialized lazily)
_local_model = None
_local_tokenizer = None

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

# HTML card for W&B — renders a side-by-side comparison with large screenshots
WANDB_CARD_TEMPLATE = """\
<div style="font-family: system-ui, -apple-system, sans-serif; width: 100%; box-sizing: border-box;">
  <div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
    <div style="font-size: 14px; color: #666; margin-bottom: 4px;">ID: {sample_id} &bull; Score: {score}/10</div>
    <div style="font-size: 16px; font-weight: 600; color: #1a1a1a;">{prompt}</div>
  </div>
  <div style="display: flex; gap: 16px; margin-bottom: 16px; width: 100%;">
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #e74c3c; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Generation</div>
      <img src="data:image/png;base64,{gen_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; display: block;" />
    </div>
    <div style="flex: 1; min-width: 0;">
      <div style="font-size: 14px; font-weight: 600; color: #27ae60; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Ground Truth</div>
      <img src="data:image/png;base64,{gt_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; display: block;" />
    </div>
  </div>
  <div style="display: flex; gap: 12px; font-size: 14px; margin-bottom: 12px;">
    <div style="flex: 1; background: #fff3f3; border-radius: 6px; padding: 12px;">
      <div style="font-weight: 600; color: #c0392b; margin-bottom: 4px;">Failure Modes</div>
      <div style="color: #333;">{failure_modes}</div>
    </div>
    <div style="flex: 1; background: #f0f7ff; border-radius: 6px; padding: 12px;">
      <div style="font-weight: 600; color: #2980b9; margin-bottom: 4px;">Code Quality</div>
      <div style="color: #333;">{code_quality}</div>
    </div>
    <div style="flex: 1; background: #f0fff4; border-radius: 6px; padding: 12px;">
      <div style="font-weight: 600; color: #27ae60; margin-bottom: 4px;">Visual Assessment</div>
      <div style="color: #333;">{visual_assessment}</div>
    </div>
  </div>
  <div style="background: #fafafa; border-radius: 6px; padding: 12px; font-size: 14px;">
    <span style="font-weight: 600; color: #555;">Reasoning:</span> {reasoning}
  </div>
</div>"""

# Failure mode taxonomy
FAILURE_MODES = [
    "generic-colors",
    "broken-layout",
    "broken-code",
    "wrong-framework",
    "no-design-thinking",
    "missing-states",
    "good",
]

JUDGE_PROMPT = """\
Rate this UI code from 1-10. Evaluate BOTH the code quality AND the visual rendering.

You will receive:
1. The code (model output vs reference)
2. Screenshots of both rendered in a real browser — use these to judge the actual visual result

Failure modes (tag all that apply):
- generic-colors: boring default palette, no intentional color scheme
- broken-layout: elements overlap, misaligned, or not responsive
- broken-code: syntax errors, blank page, or won't render
- wrong-framework: doesn't use Tailwind CSS
- no-design-thinking: works but looks like a developer prototype
- missing-states: no hover/transition/interactive polish
- good: well-designed and production-ready

CODE QUALITY: Semantic HTML, proper Tailwind classes, accessibility, responsiveness.
VISUAL RENDERING (from screenshots): Color harmony, spacing, typography hierarchy, layout balance, overall aesthetics. Compare the Generation screenshot to the GT screenshot.

User asked for: {prompt}

Generation (model output):
{model_output}

GT (ground truth):
{reference}

Respond in EXACTLY this JSON format, nothing else:
{{"score": 5, "failure_modes": ["generic-colors"], "code_quality": "brief assessment", "visual_assessment": "brief assessment of what the screenshot shows", "reasoning": "one sentence summary"}}"""


def load_test_data(test_data_path: str) -> list[dict]:
    """Load test samples from JSONL file."""
    samples = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_code(response_text: str) -> str:
    """Extract HTML/CSS code from the model response.

    Tries to find code inside triple-backtick blocks first.
    Falls back to the full response if no code blocks found.
    """
    pattern = r"```(?:html|css|tsx|jsx|vue)?\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if matches:
        return "\n".join(matches)

    stripped = response_text.strip()
    if stripped.startswith("<") or stripped.startswith("<!"):
        return stripped

    return stripped


def generate_response(client: OpenAI, model: str, question: str) -> str:
    """Send a prompt to OpenRouter and return the response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
    )
    return response.choices[0].message.content


def load_local_model(
    model_name: str,
    base_model: str = None,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    is_lora: bool = True,
):
    """Load a model using Unsloth for fast inference.
    
    Args:
        model_name: HuggingFace model ID (LoRA adapter or base model)
        base_model: Base model to load tokenizer from (for chat template). 
                    If None, uses model_name for tokenizer.
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit quantization
        is_lora: Whether this is a LoRA adapter (True) or base model (False)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _local_model, _local_tokenizer
    
    if _local_model is not None:
        return _local_model, _local_tokenizer
    
    model_type = "LoRA" if is_lora else "base"
    print(f"Loading {model_type} model: {model_name}")
    
    # Disable Unsloth statistics check to avoid timeout issues
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
    
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=not load_in_4bit,
    )
    
    # For LoRA models, the tokenizer might not have chat template
    # Load from base model if specified
    if is_lora and base_model:
        print(f"Loading tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    _local_model = model
    _local_tokenizer = tokenizer
    
    print("Model loaded successfully!")
    return model, tokenizer


# Alias for backward compatibility
def load_local_lora_model(
    lora_model: str,
    base_model: str = DEFAULT_BASE_MODEL,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
):
    """Load a LoRA model. Wrapper for load_local_model."""
    return load_local_model(
        model_name=lora_model,
        base_model=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        is_lora=True,
    )


def generate_response_local(model, tokenizer, question: str, max_new_tokens: int = 4096, use_chat_template: bool = False) -> str:
    """Generate a response using a local model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        question: The user prompt
        max_new_tokens: Maximum tokens to generate
        use_chat_template: If True, use chat template. If False, use raw text (for base models)
    
    Returns:
        The generated response text
    """
    if use_chat_template and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # For base models without chat template, use raw prompt
        text = question
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        do_sample=True,
    )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    return response


def render_html_to_screenshot(
    html_path: str,
    screenshot_path: str,
    browser,
    viewport_width: int = 1280,
    viewport_height: int = 800,
) -> bool:
    """Render an HTML file to a PNG screenshot using a Playwright browser.

    Returns True if the screenshot was captured successfully.
    """
    try:
        page = browser.new_page(
            viewport={"width": viewport_width, "height": viewport_height}
        )
        page.goto(f"file://{os.path.abspath(html_path)}", wait_until="networkidle")
        # Wait briefly for Tailwind CDN to process (if loaded from local file)
        page.wait_for_timeout(1000)
        page.screenshot(path=screenshot_path, full_page=True)
        page.close()
        return True
    except Exception as e:
        print(f"  WARNING: Screenshot failed: {e}")
        return False


def image_to_base64(image_path: str) -> str:
    """Convert a local image file to a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_to_base64_url(image_path: str) -> str:
    """Convert a local image file to a base64 data URL for multimodal API calls."""
    return f"data:image/png;base64,{image_to_base64(image_path)}"


def build_wandb_card(
    sample_id: int,
    prompt: str,
    gen_screenshot_path: str,
    gt_screenshot_path: str,
    judgment: dict,
) -> wandb.Html:
    """Build a rich HTML card for W&B with side-by-side screenshots."""
    gen_b64 = image_to_base64(gen_screenshot_path) if os.path.exists(gen_screenshot_path) else ""
    gt_b64 = image_to_base64(gt_screenshot_path) if os.path.exists(gt_screenshot_path) else ""

    html = WANDB_CARD_TEMPLATE.format(
        sample_id=sample_id,
        prompt=prompt[:300],
        score=judgment.get("score", "—"),
        gen_b64=gen_b64,
        gt_b64=gt_b64,
        failure_modes=", ".join(judgment.get("failure_modes", [])) or "—",
        code_quality=judgment.get("code_quality", "—"),
        visual_assessment=judgment.get("visual_assessment", "—"),
        reasoning=judgment.get("reasoning", "—"),
    )
    return wandb.Html(html)


def judge_output(
    judge_client: OpenAI,
    judge_model: str,
    prompt: str,
    model_output: str,
    reference: str,
    model_screenshot_path: str | None = None,
    reference_screenshot_path: str | None = None,
) -> dict:
    """Use the judge model to evaluate the generated UI code.

    Evaluates both code quality and visual rendering. When screenshots are
    available, they are sent to the multimodal judge so it can assess the
    actual rendered appearance, not just the code.
    """
    judge_text = JUDGE_PROMPT.format(
        prompt=prompt,
        model_output=model_output[:4000],  # Truncate to stay within limits
        reference=reference[:4000],
    )

    # Build multimodal message content
    content_parts = [{"type": "text", "text": judge_text}]

    if model_screenshot_path and os.path.exists(model_screenshot_path):
        content_parts.append({"type": "text", "text": "\n\nScreenshot of the GENERATION (rendered in browser):"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64_url(model_screenshot_path)},
        })

    if reference_screenshot_path and os.path.exists(reference_screenshot_path):
        content_parts.append({"type": "text", "text": "\n\nScreenshot of the GT (ground truth, rendered in browser):"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64_url(reference_screenshot_path)},
        })

    try:
        response = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": content_parts}],
            max_tokens=2048,
            temperature=0.0,
        )
        content = response.choices[0].message.content

        if not content or not content.strip():
            return {
                "score": 0,
                "failure_modes": ["judge-error"],
                "reasoning": "Judge returned empty response",
            }

        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)

        # Try to find JSON in the response (judge may include extra text)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        return {
            "score": 0,
            "failure_modes": ["judge-error"],
            "reasoning": f"Judge failed: {e}",
        }


def wrap_in_html(code: str, title: str = "UI Output") -> str:
    """Wrap extracted code in a full HTML page with Tailwind CDN."""
    if "<!DOCTYPE" in code.upper() or "<html" in code.lower():
        if "tailwindcss" not in code:
            code = code.replace(
                "<head>",
                '<head>\n  <script src="https://cdn.tailwindcss.com"></script>',
                1,
            )
        return code

    return HTML_TEMPLATE.format(title=title, content=code)


def generate_report(
    samples: list[dict],
    judgments: list[dict],
    model: str,
    judge_model: str,
    use_judge: bool,
) -> str:
    """Generate a markdown report summarizing all evaluation results."""
    lines = [
        "# UI/UX Evaluation Report",
        "",
        f"**Model**: `{model}`",
        f"**Judge**: `{judge_model}` (via OpenRouter)" if use_judge else "**Judge**: manual review",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Test samples**: {len(samples)}",
        "",
    ]

    if use_judge and judgments:
        scores = [j.get("score", 0) for j in judgments if j.get("score", 0) > 0]
        avg_score = sum(scores) / len(scores) if scores else 0
        lines.append(f"**Average Score**: {avg_score:.1f} / 10")
        lines.append("")

    lines.append("## Results")
    lines.append("")

    if use_judge:
        lines.append("| # | ID | Prompt (truncated) | Score | Failure Modes | Reasoning | Generation | GT |")
        lines.append("|---|----|--------------------|-------|---------------|-----------|------------|-----|")
    else:
        lines.append("| # | ID | Prompt (truncated) | Generation | GT | Failure Modes |")
        lines.append("|---|----|--------------------|------------|-----|---------------|")

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        question_short = sample["question"][:50].replace("|", "\\|") + "..."
        gen_img = f"![generation](screenshots/{sample_id}_generation.png)"
        gt_img = f"![gt](screenshots/{sample_id}_gt.png)"

        if use_judge and i < len(judgments):
            j = judgments[i]
            score = j.get("score", "?")
            modes = ", ".join(f"`{m}`" for m in j.get("failure_modes", []))
            reasoning = j.get("reasoning", "").replace("|", "\\|")[:80]
            lines.append(
                f"| {i+1} | {sample_id} | {question_short} | {score}/10 | {modes} | {reasoning} | {gen_img} | {gt_img} |"
            )
        else:
            lines.append(
                f"| {i+1} | {sample_id} | {question_short} | {gen_img} | {gt_img} |  |"
            )

    # Summary table
    lines.append("")
    lines.append("## Failure Mode Summary")
    lines.append("")
    lines.append("| Failure Mode | Count |")
    lines.append("|---|---|")

    if use_judge and judgments:
        mode_counts = {}
        for j in judgments:
            for mode in j.get("failure_modes", []):
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
        for mode in FAILURE_MODES:
            count = mode_counts.get(mode, 0)
            lines.append(f"| `{mode}` | {count} |")
        # Include any unexpected modes from judge
        for mode, count in mode_counts.items():
            if mode not in FAILURE_MODES:
                lines.append(f"| `{mode}` | {count} |")
    else:
        for mode in FAILURE_MODES:
            lines.append(f"| `{mode}` |  |")
        lines.append("")
        lines.append("Fill in the counts above after reviewing all outputs.")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UI/UX code generation quality"
    )
    # Model selection (mutually exclusive: OpenRouter vs Local LoRA vs Local Base)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="OpenRouter model ID (e.g. qwen/qwen3-coder-next)",
    )
    model_group.add_argument(
        "--lora-model",
        type=str,
        help="HuggingFace LoRA model ID (e.g. lilyzhng/Qwen2.5-Coder-14B-r32-20260208-164425)",
    )
    model_group.add_argument(
        "--local-model",
        type=str,
        help="HuggingFace base model ID to load locally (e.g. Qwen/Qwen2.5-Coder-14B-Instruct)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model for tokenizer when using --lora-model (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test JSONL file (e.g. data/uigen_test.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N samples (useful for quick testing)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip auto-judging, generate report for manual review only",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Skip generation, only judge existing outputs in output-dir",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenRouter model ID for the judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=WANDB_PROJECT,
        help=f"W&B project name (default: {WANDB_PROJECT})",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load LoRA model in 8-bit instead of 4-bit (uses more VRAM but more accurate)",
    )
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine model mode
    use_lora_model = args.lora_model is not None
    use_local_base_model = args.local_model is not None
    use_local_model = use_lora_model or use_local_base_model
    
    if use_lora_model:
        model_name = args.lora_model
        model_type_str = "local LoRA"
    elif use_local_base_model:
        model_name = args.local_model
        model_type_str = "local base"
    else:
        model_name = args.model
        model_type_str = "OpenRouter"
    
    # Setup OpenRouter client (always needed for judging, optionally for generation)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_client = None
    
    if not use_local_model or not args.no_judge:
        if not openrouter_key:
            if use_local_model and args.no_judge:
                print("Note: No OPENROUTER_API_KEY set, but not needed for local model without judge.")
            else:
                print("ERROR: OPENROUTER_API_KEY not set in .env.")
                print("  Add your key to .env: OPENROUTER_API_KEY=your-key-here")
                if not use_local_model:
                    sys.exit(1)
        else:
            openrouter_client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=openrouter_key,
            )

    # Load local model if using LoRA or local base model
    local_model = None
    local_tokenizer = None
    if use_lora_model:
        local_model, local_tokenizer = load_local_model(
            model_name=args.lora_model,
            base_model=args.base_model,
            load_in_4bit=not args.load_in_8bit,
            is_lora=True,
        )
    elif use_local_base_model:
        local_model, local_tokenizer = load_local_model(
            model_name=args.local_model,
            base_model=None,  # Use model's own tokenizer
            load_in_4bit=not args.load_in_8bit,
            is_lora=False,
        )

    use_judge = not args.no_judge
    if use_judge:
        if not openrouter_client:
            print("WARNING: No OpenRouter client available for judging. Disabling judge.")
            use_judge = False
        else:
            print(f"Judge model: {args.judge_model} (via OpenRouter)")

    # Load test data
    samples = load_test_data(args.test_data)
    total_available = len(samples)
    if args.limit:
        samples = samples[: args.limit]
    print(f"Evaluating {len(samples)} of {total_available} test samples from {args.test_data}")
    print(f"Model: {model_name} ({model_type_str})")
    print(f"Output: {args.output_dir}")
    print()

    # Initialize W&B run — names the run after the model for easy comparison
    model_short = model_name.split("/")[-1]
    run_name = f"{model_short}-{time.strftime('%m%d-%H%M')}"
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model": model_name,
            "model_type": model_type_str,
            "base_model": args.base_model if use_lora_model else None,
            "judge_model": args.judge_model if use_judge else "none",
            "test_data": args.test_data,
            "num_samples": len(samples),
            "total_available": total_available,
            "judge_only": args.judge_only,
        },
    )
    print(f"W&B run: {run.url}")
    print()

    # W&B: per-sample HTML cards for visual comparison

    # Launch headless browser for rendering HTML to screenshots
    print("Launching headless browser for screenshot capture...")
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    screenshots_dir = os.path.join(args.output_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    # Process each test sample
    judgments = []
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        question = sample["question"]
        reference_answer = sample["answer"]

        print(f"[{i+1}/{len(samples)}] ID={sample_id}: {question[:60]}...")

        raw_txt_path = os.path.join(args.output_dir, f"{sample_id}_raw.txt")
        output_html_path = os.path.join(args.output_dir, f"{sample_id}_generation.html")
        ref_html_path = os.path.join(args.output_dir, f"{sample_id}_gt.html")
        gen_screenshot_path = os.path.join(screenshots_dir, f"{sample_id}_generation.png")
        gt_screenshot_path = os.path.join(screenshots_dir, f"{sample_id}_gt.png")

        if args.judge_only and os.path.exists(raw_txt_path):
            # Load existing output instead of regenerating
            with open(raw_txt_path, "r", encoding="utf-8") as f:
                raw_response = f.read()
            print(f"  Loaded existing output from {raw_txt_path}")
        else:
            # Generate model response
            try:
                if use_local_model:
                    # Base models don't have chat template, use raw prompt
                    raw_response = generate_response_local(local_model, local_tokenizer, question, use_chat_template=False)
                else:
                    raw_response = generate_response(openrouter_client, args.model, question)
            except Exception as e:
                print(f"  ERROR generating: {e}")
                raw_response = f"ERROR: {e}"

            # Save raw response
            with open(raw_txt_path, "w", encoding="utf-8") as f:
                f.write(raw_response)

        # Extract code and wrap in HTML
        extracted_code = extract_code(raw_response)
        model_html = wrap_in_html(extracted_code, title=f"Generation - {sample_id}")
        reference_html = wrap_in_html(
            reference_answer, title=f"GT - {sample_id}"
        )

        # Save HTML files
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(model_html)
        with open(ref_html_path, "w", encoding="utf-8") as f:
            f.write(reference_html)

        if not args.judge_only:
            print(f"  Saved: {output_html_path}")

        # Render screenshots (for both human review and judge input)
        print("  Capturing screenshots...")
        render_html_to_screenshot(output_html_path, gen_screenshot_path, browser)
        render_html_to_screenshot(ref_html_path, gt_screenshot_path, browser)
        print(f"  Screenshots: {gen_screenshot_path}")

        # Judge the output
        judgment = {}
        if use_judge:
            print("  Judging (with visual rendering)...")
            judgment = judge_output(
                openrouter_client,
                args.judge_model,
                question,
                extracted_code,
                reference_answer,
                model_screenshot_path=gen_screenshot_path,
                reference_screenshot_path=gt_screenshot_path,
            )
            judgments.append(judgment)
            score = judgment.get("score", "?")
            modes = ", ".join(judgment.get("failure_modes", []))
            reasoning = judgment.get("reasoning", "")
            print(f"  Score: {score}/10 | Modes: {modes}")
            print(f"  Reason: {reasoning}")

            # Save judgment locally
            judgment_path = os.path.join(args.output_dir, f"{sample_id}_judgment.json")
            with open(judgment_path, "w", encoding="utf-8") as f:
                json.dump(judgment, f, indent=2)

            # Log per-sample visual comparison card to W&B
            card = build_wandb_card(
                sample_id, question,
                gen_screenshot_path, gt_screenshot_path,
                judgment,
            )
            wandb.log({
                f"samples/{sample_id}": card,
                "score": judgment.get("score", 0),
                "sample_idx": i,
            })


    # Cleanup browser
    browser.close()
    pw.stop()


    # Log summary metrics
    if use_judge and judgments:
        scores = [j.get("score", 0) for j in judgments if j.get("score", 0) > 0]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Failure mode counts
        mode_counts = {}
        for j in judgments:
            for mode in j.get("failure_modes", []):
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

        wandb.summary["avg_score"] = round(avg_score, 2)
        wandb.summary["num_samples"] = len(samples)
        wandb.summary["num_judged"] = len(judgments)
        for mode in FAILURE_MODES:
            wandb.summary[f"failure/{mode}"] = mode_counts.get(mode, 0)
        # Include unexpected modes
        for mode, count in mode_counts.items():
            if mode not in FAILURE_MODES:
                wandb.summary[f"failure/{mode}"] = count

    # Generate local report
    report = generate_report(samples, judgments, model_name, args.judge_model, use_judge)
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Print summary
    if use_judge and judgments:
        scores = [j.get("score", 0) for j in judgments if j.get("score", 0) > 0]
        if scores:
            print(f"\n{'='*50}")
            print(f"SUMMARY: Average Score = {sum(scores)/len(scores):.1f} / 10")
            print(f"{'='*50}")

    print(f"\nW&B run: {run.url}")
    wandb.finish()
    print(f"Done! Review results in {args.output_dir}/ or on W&B.")


if __name__ == "__main__":
    main()
