"""
Local evaluation script to test the judge API with OpenRouter.

This script tests the judge prompt locally without needing Modal,
useful for debugging judge response parsing issues.

Usage:
    # Set your OpenRouter API key
    export OPENROUTER_API_KEY=your_key_here
    
    # Run the script
    python finetuning/local_eval.py
    
    # Test with a specific results folder
    python finetuning/local_eval.py --results-dir wandb/eval_results/comparison-...
"""

import argparse
import json
import os
import re
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "google/gemini-3-pro-preview"  # Faster, cheaper for testing


def image_to_base64(path: str) -> str:
    """Convert a local image file to a base64 string."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def build_judge_content(
    prompt: str,
    model_output: str,
    reference: str,
    gen_img: str = None,
    gt_img: str = None,
) -> list[dict]:
    """Build content parts for the judge API message.

    Structure: text (intro + GENERATION_IMAGE label) → image → text (GROUND_TRUTH_IMAGE label) → image → text (instructions).
    """
    # 10k chars each: typical UI HTML is 6–8k; avoid truncating scripts/closing tags
    model_output_text = model_output[:10000]
    reference_text = reference[:10000]

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

    content_parts = [{"type": "text", "text": part_before_gen_image}]

    if gen_img and os.path.exists(gen_img):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_to_base64(gen_img)}"},
        })

    part_between_images = """

GROUND_TRUTH_IMAGE:
Rendered screenshot of the GROUND TRUTH reference above. Use this to compare against.
"""

    content_parts.append({"type": "text", "text": part_between_images})

    if gt_img and os.path.exists(gt_img):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_to_base64(gt_img)}"},
        })

    part_after_images = """

IMPORTANT: You MUST respond with ONLY a valid JSON object. No other text before or after.
Do not explain your reasoning outside the JSON. Put all reasoning inside the "reasoning" field.

```json
{"score": <1-10>, "failure_modes": ["<mode1>", "<mode2>"], "reasoning": "<brief explanation with penalty math>"}
```
"""
    content_parts.append({"type": "text", "text": part_after_images})

    return content_parts


def judge_output(client, judge_model: str, prompt: str, model_output: str, 
                 reference: str, gen_img: str = None, gt_img: str = None,
                 verbose: bool = True) -> dict:
    """Call the judge API and parse the response."""
    
    content_parts = build_judge_content(prompt, model_output, reference, gen_img, gt_img)

    import time
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"  Retry {attempt}/{max_retries}...")
                time.sleep(1)
            
            if verbose and attempt == 0:
                print(f"  Calling judge API: {judge_model}")

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
                    print(f"  Attempt {attempt}: Empty response")
                    last_error = "Empty response"
                    continue
            
            raw_content = raw_content.strip()
            
            if verbose:
                print(f"  Raw judge response ({len(raw_content)} chars):\n{'-'*40}")
                print(raw_content[:500] + ("..." if len(raw_content) > 500 else ""))
                print(f"{'-'*40}")
            
            # Try to parse JSON
            content = raw_content
            
            # Strip markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)
            
            # Try to find JSON object - use greedy match for nested braces
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Fix common JSON issues: replace literal newlines in strings
                # This handles cases where reasoning has newlines
                json_str = re.sub(r'(?<=: ")(.*?)(?=")', lambda m: m.group(1).replace('\n', ' '), json_str, flags=re.DOTALL)
                try:
                    parsed = json.loads(json_str)
                    if "score" in parsed:
                        if verbose:
                            print(f"  Parsed successfully: score={parsed.get('score')}")
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            # Try parsing the whole content
            try:
                parsed = json.loads(content)
                if "score" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # If we got here, no valid JSON found
            print(f"  No valid JSON found in response")
            last_error = f"No JSON: {raw_content[:100]}..."
            continue
            
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            last_error = f"JSON parse error: {e}"
            continue
        except Exception as e:
            print(f"  API error: {e}")
            last_error = f"API error: {e}"
            continue
    
    # All retries failed
    return {"score": 0, "failure_modes": ["judge-error"], "reasoning": f"After {max_retries+1} attempts: {last_error}"}


def load_sample_from_results(results_dir: Path, sample_id: str, model_type: str = "base"):
    """Load a sample from the results directory.
    
    Handles two folder structures:
    1. Old: results_dir/comparison/{files}
    2. New: results_dir/{files} (run-specific subfolder)
    """
    # Try new structure first (files directly in results_dir)
    if (results_dir / f"{sample_id}_{model_type}_raw.txt").exists():
        base_dir = results_dir
    elif (results_dir / "comparison" / f"{sample_id}_{model_type}_raw.txt").exists():
        base_dir = results_dir / "comparison"
    else:
        base_dir = results_dir  # Fall back to results_dir
    
    raw_file = base_dir / f"{sample_id}_{model_type}_raw.txt"
    gt_file = base_dir / f"{sample_id}_gt.html"
    gen_screenshot = base_dir / "screenshots" / f"{sample_id}_{model_type}.png"
    gt_screenshot = base_dir / "screenshots" / f"{sample_id}_gt.png"
    
    model_output = ""
    reference = ""
    
    if raw_file.exists():
        model_output = raw_file.read_text(encoding="utf-8")
    else:
        print(f"  Warning: {raw_file} not found")
    
    if gt_file.exists():
        reference = gt_file.read_text(encoding="utf-8")
    else:
        print(f"  Warning: {gt_file} not found")
    
    return {
        "model_output": model_output,
        "reference": reference,
        "gen_screenshot": str(gen_screenshot) if gen_screenshot.exists() else None,
        "gt_screenshot": str(gt_screenshot) if gt_screenshot.exists() else None,
    }


def test_judge_with_sample():
    """Test judge with a simple inline sample."""
    print("=" * 60)
    print("Testing judge with inline sample")
    print("=" * 60)
    
    sample_prompt = "Make a productivity timer app with minimalistic design"
    sample_output = """```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-lg shadow-lg">
        <h1 class="text-2xl font-bold mb-4">Timer</h1>
        <div class="text-6xl font-mono">25:00</div>
        <button class="mt-4 bg-blue-500 text-white px-4 py-2 rounded">Start</button>
    </div>
</body>
</html>
```"""
    
    sample_reference = """<!DOCTYPE html>
<html lang="en">
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-blue-50 flex items-center justify-center min-h-screen">
    <div class="bg-white p-12 rounded-3xl shadow-2xl">
        <h1 class="text-3xl font-semibold text-blue-800 mb-8">Productivity Timer</h1>
        <div class="relative w-48 h-48 mx-auto mb-8">
            <svg class="w-full h-full">
                <circle cx="96" cy="96" r="90" fill="none" stroke="#e2e8f0" stroke-width="8"/>
                <circle cx="96" cy="96" r="90" fill="none" stroke="#3b82f6" stroke-width="8" 
                        stroke-dasharray="565" stroke-dashoffset="141" transform="rotate(-90 96 96)"/>
            </svg>
            <div class="absolute inset-0 flex items-center justify-center text-4xl font-bold text-blue-800">25:00</div>
        </div>
        <div class="flex gap-4 justify-center">
            <button class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-full transition">Start</button>
            <button class="bg-gray-200 hover:bg-gray-300 text-gray-700 px-6 py-3 rounded-full transition">Reset</button>
        </div>
    </div>
</body>
</html>"""
    
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY=your_key_here")
        return
    
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    
    result = judge_output(
        client=client,
        judge_model=DEFAULT_JUDGE_MODEL,
        prompt=sample_prompt,
        model_output=sample_output,
        reference=sample_reference,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Final parsed result:")
    print(json.dumps(result, indent=2))


def test_judge_with_results(results_dir: Path, sample_id: str = "test_0"):
    """Test judge with actual results from a previous run."""
    print("=" * 60)
    print(f"Testing judge with results from: {results_dir}")
    print(f"Sample: {sample_id}")
    print("=" * 60)
    
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return
    
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    
    # Test base model output
    print("\n--- Testing BASE model output ---")
    base_data = load_sample_from_results(results_dir, sample_id, "base")
    if base_data["model_output"]:
        base_result = judge_output(
            client=client,
            judge_model=DEFAULT_JUDGE_MODEL,
            prompt="Make a productivity timer app with minimalistic design, circular countdowns, and calming pastel backgrounds.",
            model_output=base_data["model_output"],
            reference=base_data["reference"],
            gen_img=base_data["gen_screenshot"],
            gt_img=base_data["gt_screenshot"],
            verbose=True,
        )
        print(f"\nBase result: {json.dumps(base_result, indent=2)}")
    else:
        print("No base model output found")
    
    # Test LoRA model output
    print("\n--- Testing LORA model output ---")
    lora_data = load_sample_from_results(results_dir, sample_id, "lora")
    if lora_data["model_output"]:
        lora_result = judge_output(
            client=client,
            judge_model=DEFAULT_JUDGE_MODEL,
            prompt="Make a productivity timer app with minimalistic design, circular countdowns, and calming pastel backgrounds.",
            model_output=lora_data["model_output"],
            reference=lora_data["reference"],
            gen_img=lora_data["gen_screenshot"],
            gt_img=lora_data["gt_screenshot"],
            verbose=True,
        )
        print(f"\nLoRA result: {json.dumps(lora_result, indent=2)}")
    else:
        print("No LoRA model output found")


def main():
    global DEFAULT_JUDGE_MODEL
    
    # Get project root (parent of finetuning directory)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description="Test judge API locally")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to results directory to test with actual outputs (relative to project root or absolute)",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default="test_0",
        help="Sample ID to test (default: test_0)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model to use (default: {DEFAULT_JUDGE_MODEL})",
    )
    
    args = parser.parse_args()
    
    DEFAULT_JUDGE_MODEL = args.judge_model
    
    if args.results_dir:
        results_path = Path(args.results_dir)
        # If path is relative, resolve it relative to project root
        if not results_path.is_absolute():
            results_path = project_root / results_path
        results_path = results_path.resolve()
        
        print(f"Project root: {project_root}")
        print(f"Results path: {results_path}")
        
        if not results_path.exists():
            print(f"ERROR: Results directory not found: {results_path}")
            print(f"Available directories in wandb/eval_results/:")
            eval_results_dir = project_root / "wandb" / "eval_results"
            if eval_results_dir.exists():
                for d in eval_results_dir.iterdir():
                    if d.is_dir():
                        print(f"  - {d.name}")
            return
        test_judge_with_results(results_path, args.sample_id)
    else:
        test_judge_with_sample()


if __name__ == "__main__":
    main()
