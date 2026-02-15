"""
Evaluate UI/UX code generation on Modal with GPU.

Compares a base model against a finetuned Finetuned model on the same test data.
Both runs are logged to W&B for easy comparison.

Usage:
    # Evaluate both base and Finetuned models (default: 20 samples)
    modal run finetuning/modal_eval.py

    # Custom limit
    modal run finetuning/modal_eval.py --limit 50

    # Only evaluate Finetuned model
    modal run finetuning/modal_eval.py --lora-only

    # Only evaluate base model
    modal run finetuning/modal_eval.py --base-only

    # Skip judge (faster, no scoring)
    modal run finetuning/modal_eval.py --no-judge
"""

from dataclasses import dataclass
import modal

# ---------------------------------------------------------------------------
# Modal App & Infrastructure
# ---------------------------------------------------------------------------
app = modal.App("uiux-eval")

# Container image with all dependencies
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[cu128-torch270]",
        "datasets",
        "hf-transfer",
        "wandb",
        "openai",
        "python-dotenv",
        "playwright",
    )
    .run_commands(
        "playwright install --with-deps chromium"
    )
    .env({
        "HF_HOME": "/model_cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# Persistent volumes
model_cache_vol = modal.Volume.from_name("uiux-model-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("uiux-eval-results", create_if_missing=True)

# GPU config
GPU_CONFIG = "A100-80GB"
TIMEOUT_HOURS = 2


@dataclass
class EvalConfig:
    """Configuration for side-by-side comparison evaluation."""
    base_model: str = "Qwen/Qwen2.5-Coder-14B"
    lora_model: str = "lilyzhng/Qwen2.5-Coder-14B-r32-20260208-164425"
    hf_dataset: str = "lilyzhng/uigen-ui-code-gen"  # HuggingFace dataset
    output_base_dir: str = "/results"
    limit: int = 20
    judge_model: str = "google/gemini-3-pro-preview"
    wandb_project: str = "uiux-eval"
    use_judge: bool = True


@app.function(
    image=eval_image,
    gpu=GPU_CONFIG,
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
    """Run evaluation on both base and Finetuned models."""
    import os
    import json
    import time
    import base64
    import re

    import wandb
    from openai import OpenAI
    from playwright.sync_api import sync_playwright

    # Disable Unsloth statistics check
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer

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
    <div style="flex: 1; min-width: 0; text-align: center;">
      <div style="font-size: 14px; font-weight: 600; color: #27ae60; margin-bottom: 8px;">Ground Truth</div>
      <img src="data:image/png;base64,{gt_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
    <div style="flex: 1; min-width: 0; text-align: center;">
      <div style="font-size: 14px; font-weight: 600; color: #3498db; margin-bottom: 8px;">Base Model</div>
      <img src="data:image/png;base64,{base_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
    <div style="flex: 1; min-width: 0; text-align: center;">
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

    # ---------------------------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------------------------
    def build_judge_content(prompt, model_output, reference, gen_img, gt_img):
        """Build content parts for the judge API message.

        Structure: text (intro + GENERATION_IMAGE label) → image → text (GROUND_TRUTH_IMAGE label) → image → text (instructions).
        """
        # 10k chars each: typical UI HTML is 6–8k; avoid truncating scripts/closing tags
        model_output_text = model_output #model_output[:10000]
        reference_text = reference #reference[:10000]

        part_before_gen_image = f"""\
You are a UI code quality judge. Rate the generation from 1-10.

EVALUATION: Use the generation and ground truth screenshots to compare the rendered UI and evaluate visual quality. **First check the screenshots** to see how the UI actually renders, **then check the code** for implementation details (framework, syntax, etc.).

SCORING RUBRIC (start at 10, subtract for issues):
- broken-code (-4): syntax errors, blank page, no output
- broken-layout (-3): elements overlap, misaligned, unusable  
- wrong-framework (-2): doesn't use Tailwind CSS
- generic-colors (-2): boring default palette
- no-design-thinking (-2): looks like developer prototype
- missing-states (-1): no hover/transition polish

TASK: {prompt}

GENERATION:
{model_output_text}

GROUND TRUTH:
{reference_text}

GENERATION_IMAGE:
Rendered screenshot of the GENERATION code above. Use this to evaluate the visual output.
"""

        part_between_images = """

GROUND_TRUTH_IMAGE:
Rendered screenshot of the GROUND TRUTH reference above. Use this to compare against.
"""

        part_after_images = """

IMPORTANT: You MUST respond with ONLY a valid JSON object. No other text before or after.
Do not explain your reasoning outside the JSON. Put all reasoning inside the "reasoning" field.

```json
{"score": <1-10>, "failure_modes": ["<mode1>", "<mode2>"], "reasoning": "<brief explanation with penalty math>"}
```
"""

        content_parts = [{"type": "text", "text": part_before_gen_image}]

        gen_exists = gen_img and os.path.exists(gen_img)
        if gen_exists:
            gen_b64 = image_to_base64(gen_img)
            if gen_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{gen_b64}"},
                })

        content_parts.append({"type": "text", "text": part_between_images})

        gt_exists = gt_img and os.path.exists(gt_img)
        if gt_exists:
            gt_b64 = image_to_base64(gt_img)
            if gt_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{gt_b64}"},
                })

        content_parts.append({"type": "text", "text": part_after_images})

        return content_parts

    # Prompt template matching the training format
    PROMPT_TEMPLATE = "# Task: Generate HTML/CSS code using Tailwind CSS\n# Requirements: {requirements}\n\n"
    
    def load_test_data(hf_dataset: str) -> list[dict]:
        """Load test data from HuggingFace dataset."""
        from datasets import load_dataset
        
        dataset = load_dataset(hf_dataset, split="test")
        samples = []
        
        for i, item in enumerate(dataset):
            text = item["text"]
            # Parse the text format: "# Task: ...\n# Requirements: ...\n```html\n...\n```"
            # Extract requirements and answer (code)
            
            # Find the requirements line
            lines = text.split("\n")
            requirements = ""
            for line in lines:
                if line.startswith("# Requirements:"):
                    requirements = line.replace("# Requirements:", "").strip()
                    break
            
            # Build the full prompt matching training format
            # This is critical for base models - they need the exact same prompt format
            full_prompt = PROMPT_TEMPLATE.format(requirements=requirements)
            
            # Extract the code (answer) from the markdown code block
            code_match = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.DOTALL)
            answer = code_match.group(1).strip() if code_match else ""
            
            samples.append({
                "id": f"test_{i}",
                "question": full_prompt,  # Use full prompt, not just requirements
                "requirements": requirements,  # Keep original for display
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
        content_parts = build_judge_content(prompt, model_output, reference, gen_img, gt_img)
        gen_exists = gen_img and os.path.exists(gen_img)
        gt_exists = gt_img and os.path.exists(gt_img)
        print(f"    Images: gen={gen_exists}, gt={gt_exists}")

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"    Retry {attempt}/{max_retries}...")
                    time.sleep(1)  # Brief pause before retry

                # Gemini 3 requires reasoning and cannot disable it. Use large max_tokens so
                # the model has room for both reasoning tokens and the final JSON output.
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": content_parts}],
                    max_tokens=8192,  # Room for reasoning + final output (Gemini 3 mandates reasoning)
                    temperature=0.0,
                )
                raw_content = response.choices[0].message.content
                if raw_content is None or len((raw_content or "").strip()) == 0:
                    # Fallback: some reasoning models put output in reasoning_details
                    msg = response.choices[0].message
                    reasoning = getattr(msg, "reasoning", None) or ""
                    if not reasoning and hasattr(msg, "reasoning_details") and msg.reasoning_details:
                        for rd in msg.reasoning_details:
                            if getattr(rd, "text", None):
                                reasoning += rd.text or ""
                    if reasoning and "score" in reasoning:
                        raw_content = reasoning
                    else:
                        print(f"    Attempt {attempt}: Empty response")
                        last_error = "Empty response"
                        continue
                    
                raw_content = raw_content.strip()
                print(f"    Response length: {len(raw_content)}, first 300 chars: {raw_content[:300]}")
                
                # Strip markdown code blocks
                content = raw_content
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                    content = re.sub(r"\n?```\s*$", "", content)
                
                # Find JSON object - handle nested braces for failure_modes array
                json_match = re.search(r"\{[^{}]*(?:\[[^\[\]]*\][^{}]*)*\}", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    # Fix newlines in string values (common issue with reasoning field)
                    json_str = re.sub(r'(?<=: ")(.*?)(?=")', lambda m: m.group(1).replace('\n', ' ').replace('\r', ''), json_str, flags=re.DOTALL)
                    try:
                        parsed = json.loads(json_str)
                        if "score" in parsed:
                            print(f"    Parsed score: {parsed.get('score')}")
                            return parsed
                        else:
                            print(f"    JSON missing 'score' field: {parsed}")
                            last_error = "Missing score field"
                            continue
                    except json.JSONDecodeError as e:
                        print(f"    JSON parse failed after cleanup: {e}")
                        last_error = f"JSON parse error: {e}"
                        continue
                
                # Try parsing the whole content as JSON
                try:
                    parsed = json.loads(content)
                    if "score" in parsed:
                        print(f"    Parsed score: {parsed.get('score')}")
                        return parsed
                except json.JSONDecodeError:
                    pass
                
                # If we got text but no valid JSON, the model didn't follow format
                print(f"    No valid JSON found in response")
                last_error = f"No JSON in response: {raw_content[:100]}..."
                continue
                
            except json.JSONDecodeError as e:
                print(f"    JSON parse error: {e}")
                last_error = f"JSON parse error: {e}"
                continue
            except Exception as e:
                print(f"    API error: {type(e).__name__}: {e}")
                last_error = f"API error: {e}"
                continue
        
        # All retries failed
        return {"score": 0, "failure_modes": ["judge-error"], "reasoning": f"After {max_retries+1} attempts: {last_error}"}

    def generate_response(model, tokenizer, question: str, max_new_tokens: int = 4096, use_chat_template: bool = False) -> str:
        """Generate response from model.
        
        Args:
            model: The loaded model
            tokenizer: The tokenizer
            question: The input prompt
            max_new_tokens: Maximum tokens to generate
            use_chat_template: If True, use chat template. If False, use raw text (for base models)
        """
        if use_chat_template and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run_comparison_eval(
        base_model,
        base_tokenizer,
        lora_model,
        lora_tokenizer,
        samples: list[dict],
        output_base_dir: str,
        openrouter_client,
        config: EvalConfig,
    ) -> dict:
        """Run side-by-side evaluation comparing base and Finetuned models."""
        # Initialize W&B with comparison run
        base_short = config.base_model.split("/")[-1]
        lora_short = config.lora_model.split("/")[-1]
        run_name = f"comparison-{base_short}-vs-{lora_short}-{time.strftime('%m%d-%H%M')}"
        
        # Create run-specific output directory to avoid accumulating results
        output_dir = os.path.join(output_base_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        screenshots_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "base_model": config.base_model,
                "lora_model": config.lora_model,
                "judge_model": config.judge_model if config.use_judge else "none",
                "num_samples": len(samples),
            },
        )
        print(f"W&B run: {run.url}")

        # Launch browser
        print("Launching browser...")
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)

        base_judgments = []
        lora_judgments = []
        
        for i, sample in enumerate(samples):
            sample_id = sample["id"]
            question = sample["question"]  # Full prompt with template
            requirements = sample.get("requirements", question[:100])  # For display
            reference = sample["answer"]

            print(f"\n[{i+1}/{len(samples)}] ID={sample_id}: {requirements[:50]}...")

            # Generate from BASE model
            print("  Generating from base model...")
            try:
                base_raw = generate_response(base_model, base_tokenizer, question, use_chat_template=False)
            except Exception as e:
                print(f"  BASE ERROR: {e}")
                base_raw = f"ERROR: {e}"

            # Generate from Finetuned model
            print("  Generating from Finetuned model...")
            try:
                lora_raw = generate_response(lora_model, lora_tokenizer, question, use_chat_template=False)
            except Exception as e:
                print(f"  LORA ERROR: {e}")
                lora_raw = f"ERROR: {e}"

            # Process HTML
            base_extracted = extract_code(base_raw)
            lora_extracted = extract_code(lora_raw)
            
            base_html = wrap_in_html(base_extracted, f"Base-{sample_id}")
            lora_html = wrap_in_html(lora_extracted, f"Finetuned-{sample_id}")
            gt_html = wrap_in_html(reference, f"GT-{sample_id}")

            # Save files
            base_path = os.path.join(output_dir, f"{sample_id}_base.html")
            lora_path = os.path.join(output_dir, f"{sample_id}_lora.html")
            gt_path = os.path.join(output_dir, f"{sample_id}_gt.html")
            
            base_img = os.path.join(screenshots_dir, f"{sample_id}_base.png")
            lora_img = os.path.join(screenshots_dir, f"{sample_id}_lora.png")
            gt_img = os.path.join(screenshots_dir, f"{sample_id}_gt.png")

            with open(base_path, "w") as f:
                f.write(base_html)
            with open(lora_path, "w") as f:
                f.write(lora_html)
            with open(gt_path, "w") as f:
                f.write(gt_html)
            with open(os.path.join(output_dir, f"{sample_id}_base_raw.txt"), "w") as f:
                f.write(base_raw)
            with open(os.path.join(output_dir, f"{sample_id}_lora_raw.txt"), "w") as f:
                f.write(lora_raw)

            # Screenshots
            print("  Rendering screenshots...")
            base_ss_ok = render_screenshot(base_path, base_img, browser)
            lora_ss_ok = render_screenshot(lora_path, lora_img, browser)
            gt_ss_ok = render_screenshot(gt_path, gt_img, browser)
            print(f"  Screenshot results: base={base_ss_ok}, lora={lora_ss_ok}, gt={gt_ss_ok}")

            # Judge both outputs
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

                print("  Judging Finetuned model output...")
                lora_judgment = judge_output(
                    openrouter_client, config.judge_model,
                    question, lora_extracted, reference, lora_img, gt_img
                )
                lora_judgments.append(lora_judgment)
                print(f"  Finetuned Score: {lora_judgment.get('score', '?')}/10")

                # Log 3-column comparison to W&B
                card_html = WANDB_CARD_TEMPLATE.format(
                    sample_id=sample_id,
                    prompt=requirements[:200],  # Show requirements, not full prompt
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

                # Save judgments
                with open(os.path.join(output_dir, f"{sample_id}_base_judgment.json"), "w") as f:
                    json.dump(base_judgment, f, indent=2)
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
        
        wandb.summary["score_improvement"] = round(lora_avg - base_avg, 2)
        wandb.summary["num_samples"] = len(samples)
        
        print(f"\n{'='*40}")
        print(f"Base Model Avg Score: {base_avg:.1f}/10")
        print(f"Finetuned Model Avg Score: {lora_avg:.1f}/10")
        print(f"Improvement: {lora_avg - base_avg:+.1f}")
        print(f"{'='*40}")

        wandb.finish()
        return {
            "base_avg_score": base_avg if base_judgments else None,
            "lora_avg_score": lora_avg if lora_judgments else None,
            "run_url": run.url,
            "run_name": run_name,  # Include run name for local download folder
        }

    # ---------------------------------------------------------------------------
    # Main Execution
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("UIUX Evaluation on Modal - Side-by-Side Comparison")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"Finetuned model: {config.lora_model}")
    print(f"HF Dataset: {config.hf_dataset}")
    print(f"Limit: {config.limit}")
    print(f"Use judge: {config.use_judge}")
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
            # Mask key for logging (show first 8 and last 4 chars)
            masked_key = f"{openrouter_key[:8]}...{openrouter_key[-4:]}" if len(openrouter_key) > 12 else "***"
            print(f"OpenRouter API key found: {masked_key}")
            
            openrouter_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key)
            print(f"Judge: {config.judge_model}")
            
            # Test the API connection with a simple request
            print("Testing OpenRouter API connection...")
            try:
                test_response = openrouter_client.chat.completions.create(
                    model=config.judge_model,
                    messages=[{"role": "user", "content": "Reply with just the word 'OK'"}],
                    max_tokens=10,
                    temperature=0.0,
                )
                test_content = test_response.choices[0].message.content.strip()
                print(f"  API test successful! Response: {test_content}")
            except Exception as e:
                print(f"  WARNING: API test failed: {e}")
                print("  Continuing anyway, but judge calls may fail...")
        else:
            print("WARNING: No OPENROUTER_API_KEY found in environment, disabling judge")
            print("  Available env vars:", [k for k in os.environ.keys() if 'KEY' in k or 'SECRET' in k or 'TOKEN' in k])
            config.use_judge = False

    # ---------------------------------------------------------------------------
    # Load Both Models for Side-by-Side Comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    
    # Load base model
    print(f"Loading base model: {config.base_model}")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)
    print("Base model loaded!")

    # Load Finetuned model
    print(f"Loading Finetuned model: {config.lora_model}")
    lora_model, _ = FastLanguageModel.from_pretrained(
        model_name=config.lora_model,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    # Load tokenizer from base model (Finetuned was trained on base model)
    lora_tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    FastLanguageModel.for_inference(lora_model)
    print("Finetuned model loaded!")

    # ---------------------------------------------------------------------------
    # Run Side-by-Side Comparison Evaluation
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    
    # Pass the base output directory - run_comparison_eval will create run-specific subfolder
    results = run_comparison_eval(
        base_model, base_tokenizer,
        lora_model, lora_tokenizer,
        samples, config.output_base_dir,
        openrouter_client, config
    )

    # ---------------------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    if results.get("base_avg_score") is not None:
        print(f"Base Model Score: {results['base_avg_score']:.1f}/10")
    
    if results.get("lora_avg_score") is not None:
        print(f"Finetuned Model Score: {results['lora_avg_score']:.1f}/10")
    
    if results.get("base_avg_score") is not None and results.get("lora_avg_score") is not None:
        diff = results["lora_avg_score"] - results["base_avg_score"]
        print(f"Improvement: {diff:+.1f} points")
    
    print(f"W&B: {results.get('run_url', 'N/A')}")

    # Commit results to volume
    results_vol.commit()
    
    return results


@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install("openai"),
    timeout=120,
    secrets=[modal.Secret.from_name("openrouter-secret")],
)
def test_openrouter_api(judge_model: str = "google/gemini-3-pro-preview"):
    """Test OpenRouter API connection from Modal.
    
    Run with: modal run finetuning/modal_eval.py::test_openrouter_api
    """
    import os
    from openai import OpenAI
    
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    print("=" * 60)
    print("Testing OpenRouter API from Modal")
    print("=" * 60)
    
    # Check environment variables
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    print(f"\nEnvironment variables with KEY/SECRET/TOKEN:")
    for k in sorted(os.environ.keys()):
        if any(x in k.upper() for x in ['KEY', 'SECRET', 'TOKEN', 'API']):
            v = os.environ[k]
            masked = f"{v[:8]}...{v[-4:]}" if len(v) > 12 else "***"
            print(f"  {k}: {masked}")
    
    if not openrouter_key:
        print("\nERROR: OPENROUTER_API_KEY not found!")
        print("Make sure the modal secret 'openrouter-secret' contains OPENROUTER_API_KEY")
        return {"success": False, "error": "No API key"}
    
    print(f"\nAPI key found: {openrouter_key[:8]}...{openrouter_key[-4:]}")
    print(f"Judge model: {judge_model}")
    
    # Test API
    print("\nTesting API connection...")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key)
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        print(f"Response: {content}")
        print("\nSUCCESS: OpenRouter API is working!")
        return {"success": True, "response": content}
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        return {"success": False, "error": str(e)}


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": results_vol},
    timeout=600,
)
def list_results(run_name: str = None):
    """List files in the results volume.
    
    Args:
        run_name: If provided, only list files from this run's subfolder.
                  If None, list all files (legacy behavior).
    """
    import os
    
    if run_name:
        base_path = f"/results/{run_name}"
    else:
        base_path = "/results"
    
    if not os.path.exists(base_path):
        print(f"Path does not exist: {base_path}")
        return []
    
    files = []
    for root, dirs, filenames in os.walk(base_path):
        for f in filenames:
            path = os.path.join(root, f)
            files.append(path)
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
    base_model: str = "Qwen/Qwen2.5-Coder-14B",
    lora_model: str = "lilyzhng/Qwen2.5-Coder-14B-r32-20260208-164425",
    download_only: bool = False,
    local_output: str = "wandb/eval_results",
):
    """Run UIUX side-by-side comparison evaluation on Modal.
    
    This will generate a 3-column comparison in W&B:
    - Column 1: Ground Truth
    - Column 2: Base Model Generation
    - Column 3: Finetuned Model Generation
    
    Results are downloaded to a subfolder named after the W&B run name
    to avoid overwriting previous evaluations.
    
    Args:
        limit: Number of samples to evaluate
        no_judge: Skip LLM judging
        base_model: Base model to compare
        lora_model: Finetuned model to compare
        download_only: Only download existing results (skip evaluation)
        local_output: Base directory to save results (subfolder created per run)
    """
    import os
    import time
    
    run_name = None
    
    if not download_only:
        config = EvalConfig(
            base_model=base_model,
            lora_model=lora_model,
            limit=limit,
            use_judge=not no_judge,
        )
        
        print("Starting UIUX side-by-side comparison evaluation on Modal...")
        print(f"  Base model: {config.base_model}")
        print(f"  Finetuned model: {config.lora_model}")
        print(f"  Limit: {config.limit}")
        print(f"  Use judge: {config.use_judge}")
        
        results = run_evaluation.remote(config)
        print("\nResults:", results)
        
        # Get run name from results for subfolder
        run_name = results.get("run_name")
    
    # Create subfolder name: use run_name if available, otherwise timestamp
    if run_name:
        subfolder = run_name
    else:
        subfolder = f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    
    output_dir = os.path.join(local_output, subfolder)
    
    # Download results from Modal volume to local
    print(f"\nDownloading results to {output_dir}/...")
    
    # Only download files from the current run's folder
    files = list_results.remote(run_name=run_name)
    print(f"Found {len(files)} files for run '{run_name}' in Modal volume")
    
    # Create local directory
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    for remote_path in files:
        # Convert /results/{run_name}/... to local path (strip /results/{run_name}/ prefix)
        if run_name:
            relative_path = remote_path.replace(f"/results/{run_name}/", "")
        else:
            relative_path = remote_path.replace("/results/", "")
        local_path = os.path.join(output_dir, relative_path)
        
        # Create parent directories
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        try:
            content = read_file.remote(remote_path)
            with open(local_path, "wb") as f:
                f.write(content)
            downloaded += 1
            print(f"  Downloaded: {relative_path}")
        except Exception as e:
            print(f"  Failed: {relative_path} - {e}")
    
    print(f"\nDownloaded {downloaded}/{len(files)} files to {output_dir}/")
    print("You can now view the generated HTML/code locally!")
