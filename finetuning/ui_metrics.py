"""UI metrics utilities for training callbacks."""

from collections import Counter
from html.parser import HTMLParser
from typing import Dict, List, Optional

import numpy as np
import re

try:
    from transformers import TrainerCallback
except Exception:
    TrainerCallback = object


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
    classes: List[str] = []
    for pattern in class_patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            classes.extend(match.split())
    return classes


def extract_color_tokens(code: str) -> List[str]:
    """Extract color-related tokens (Tailwind colors and hex codes)."""
    colors: List[str] = []
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
        import wandb

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
        """Generate samples and compute UI code metrics."""
        import torch
        import wandb

        model.eval()

        all_colors: List[str] = []
        all_classes: List[str] = []
        valid_syntax_count = 0
        sample_generations = []

        # Generate from first validation sample only (to save time during training)
        sample = self.val_samples[0]
        prompt_text = (
            f"<|im_start|>user\n{sample.get('question', '')}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        ground_truth = sample.get("answer", "")

        # Generate (best-effort; avoid torch.compile issues during generate)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                with torch._dynamo.disable():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,  # Longer for better UI generation
                        temperature=0.7,
                        do_sample=True,
                    )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = generated[len(prompt_text):].strip()
        except Exception as exc:
            print(f"[UI Metrics] Generation failed at step {step}: {exc}")
            wandb.log({"eval/generation_error": 1}, step=step)
            return

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

        # Save for W&B table
        sample_generations.append(
            {
                "step": step,
                "prompt": sample.get("question", "")[:100] + "...",
                "ground_truth": ground_truth[:400] + "..." if len(ground_truth) > 400 else ground_truth,
                "generated_code": (
                    generated_code[:400] + "..." if len(generated_code) > 400 else generated_code
                ),
                "syntax_valid": "✓" if is_valid else "✗",
                "num_colors": len(colors),
                "num_classes": len(classes),
            }
        )

        # Aggregate metrics
        syntax_validity_rate = valid_syntax_count / 1  # Just 1 sample
        color_entropy = compute_color_entropy(all_colors)
        unique_classes = len(set(all_classes))

        # Log to W&B
        wandb.log(
            {
                "eval/syntax_validity_rate": syntax_validity_rate,
                "eval/color_entropy": color_entropy,
                "eval/unique_tailwind_classes": unique_classes,
                "eval/total_colors_used": len(all_colors),
            },
            step=step,
        )

        # Log sample generations as table
        if sample_generations:
            wandb.log(
                {
                    "eval/sample_details": wandb.Table(
                        columns=list(sample_generations[0].keys()),
                        data=[list(s.values()) for s in sample_generations],
                    )
                },
                step=step,
            )

        model.train()
