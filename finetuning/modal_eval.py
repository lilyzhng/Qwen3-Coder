"""
Evaluate UI/UX code generation on Modal with GPU.

Compares a base model against a finetuned LoRA model on the same test data.
Both runs are logged to W&B for easy comparison.

Usage:
    # Evaluate both base and LoRA models (default: 20 samples)
    modal run finetuning/modal_eval.py

    # Custom limit
    modal run finetuning/modal_eval.py --limit 50

    # Only evaluate LoRA model
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
    """Run evaluation on both base and LoRA models."""
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

    # Card template for 3-column comparison: Ground Truth | Base Generation | LoRA Generation
    WANDB_CARD_TEMPLATE = """\
<div style="font-family: system-ui, -apple-system, sans-serif; width: 100%; box-sizing: border-box;">
  <div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
    <div style="font-size: 14px; color: #666; margin-bottom: 4px;">ID: {sample_id} &bull; Base Score: {base_score}/10 &bull; LoRA Score: {lora_score}/10</div>
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
      <div style="font-size: 14px; font-weight: 600; color: #9b59b6; margin-bottom: 8px;">LoRA Model</div>
      <img src="data:image/png;base64,{lora_b64}" style="width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" />
    </div>
  </div>
  <div style="background: #fafafa; border-radius: 6px; padding: 12px; font-size: 14px;">
    <div style="margin-bottom: 8px;">
      <span style="font-weight: 600; color: #3498db;">Base Failure Modes:</span> {base_failure_modes}<br/>
      <span style="font-weight: 600; color: #3498db;">Base Reasoning:</span> {base_reasoning}
    </div>
    <div>
      <span style="font-weight: 600; color: #9b59b6;">LoRA Failure Modes:</span> {lora_failure_modes}<br/>
      <span style="font-weight: 600; color: #9b59b6;">LoRA Reasoning:</span> {lora_reasoning}
    </div>
  </div>
</div>"""

    JUDGE_PROMPT = """\
Rate this UI code from 1-10. Evaluate BOTH the code quality AND the visual rendering.

Failure modes (tag all that apply):
- generic-colors: boring default palette
- broken-layout: elements overlap, misaligned
- broken-code: syntax errors, blank page
- wrong-framework: doesn't use Tailwind CSS
- no-design-thinking: works but looks like a developer prototype
- missing-states: no hover/transition polish
- good: well-designed and production-ready

User asked for: {prompt}

Generation (model output):
{model_output}

GT (ground truth):
{reference}

Respond in EXACTLY this JSON format:
{{"score": 5, "failure_modes": ["generic-colors"], "reasoning": "one sentence summary"}}"""

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

    def judge_output(client, judge_model, prompt, model_output, reference, gen_img, gt_img) -> dict:
        judge_text = JUDGE_PROMPT.format(
            prompt=prompt,
            model_output=model_output[:4000],
            reference=reference[:4000],
        )
        content_parts = [{"type": "text", "text": judge_text}]
        
        if os.path.exists(gen_img):
            content_parts.append({"type": "text", "text": "\n\nGeneration screenshot:"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_to_base64(gen_img)}"},
            })
        if os.path.exists(gt_img):
            content_parts.append({"type": "text", "text": "\n\nGround truth screenshot:"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_to_base64(gt_img)}"},
            })

        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": content_parts}],
                max_tokens=1024,
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except Exception as e:
            return {"score": 0, "failure_modes": ["judge-error"], "reasoning": f"Error: {e}"}

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
        output_dir: str,
        openrouter_client,
        config: EvalConfig,
    ) -> dict:
        """Run side-by-side evaluation comparing base and LoRA models."""
        os.makedirs(output_dir, exist_ok=True)
        screenshots_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        # Initialize W&B with comparison run
        base_short = config.base_model.split("/")[-1]
        lora_short = config.lora_model.split("/")[-1]
        run_name = f"comparison-{base_short}-vs-{lora_short}-{time.strftime('%m%d-%H%M')}"
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

            # Generate from LoRA model
            print("  Generating from LoRA model...")
            try:
                lora_raw = generate_response(lora_model, lora_tokenizer, question, use_chat_template=False)
            except Exception as e:
                print(f"  LORA ERROR: {e}")
                lora_raw = f"ERROR: {e}"

            # Process HTML
            base_extracted = extract_code(base_raw)
            lora_extracted = extract_code(lora_raw)
            
            base_html = wrap_in_html(base_extracted, f"Base-{sample_id}")
            lora_html = wrap_in_html(lora_extracted, f"LoRA-{sample_id}")
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
            render_screenshot(base_path, base_img, browser)
            render_screenshot(lora_path, lora_img, browser)
            render_screenshot(gt_path, gt_img, browser)

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

                print("  Judging LoRA model output...")
                lora_judgment = judge_output(
                    openrouter_client, config.judge_model,
                    question, lora_extracted, reference, lora_img, gt_img
                )
                lora_judgments.append(lora_judgment)
                print(f"  LoRA Score: {lora_judgment.get('score', '?')}/10")

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
        print(f"LoRA Model Avg Score: {lora_avg:.1f}/10")
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
    print(f"LoRA model: {config.lora_model}")
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
            openrouter_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key)
            print(f"Judge: {config.judge_model}")
        else:
            print("WARNING: No OPENROUTER_API_KEY, disabling judge")
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

    # Load LoRA model
    print(f"Loading LoRA model: {config.lora_model}")
    lora_model, _ = FastLanguageModel.from_pretrained(
        model_name=config.lora_model,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    # Load tokenizer from base model (LoRA was trained on base model)
    lora_tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    FastLanguageModel.for_inference(lora_model)
    print("LoRA model loaded!")

    # ---------------------------------------------------------------------------
    # Run Side-by-Side Comparison Evaluation
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    
    output_dir = os.path.join(config.output_base_dir, "comparison")
    results = run_comparison_eval(
        base_model, base_tokenizer,
        lora_model, lora_tokenizer,
        samples, output_dir,
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
        print(f"LoRA Model Score: {results['lora_avg_score']:.1f}/10")
    
    if results.get("base_avg_score") is not None and results.get("lora_avg_score") is not None:
        diff = results["lora_avg_score"] - results["base_avg_score"]
        print(f"Improvement: {diff:+.1f} points")
    
    print(f"W&B: {results.get('run_url', 'N/A')}")

    # Commit results to volume
    results_vol.commit()
    
    return results


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/results": results_vol},
    timeout=600,
)
def list_results():
    """List all files in the results volume."""
    import os
    files = []
    for root, dirs, filenames in os.walk("/results"):
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
    - Column 3: LoRA Model Generation
    
    Results are downloaded to a subfolder named after the W&B run name
    to avoid overwriting previous evaluations.
    
    Args:
        limit: Number of samples to evaluate
        no_judge: Skip LLM judging
        base_model: Base model to compare
        lora_model: LoRA model to compare
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
        print(f"  LoRA model: {config.lora_model}")
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
    
    files = list_results.remote()
    print(f"Found {len(files)} files in Modal volume")
    
    # Create local directory
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    for remote_path in files:
        # Convert /results/comparison/... to local path
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
