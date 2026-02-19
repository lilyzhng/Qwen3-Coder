"""
Custom metrics and callbacks for monitoring UI/UX code generation during SFT.

Based on: SFT Monitoring Guide for UIGEN Dataset
Focus: UI Code Integrity (no reasoning traces, only code output)
"""

import re
from collections import Counter
from typing import Dict, List, Optional
import numpy as np
import torch
import wandb
from transformers import TrainerCallback
from html.parser import HTMLParser


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


def compute_syntax_validity(html_code: str) -> bool:
    """Check if HTML/JSX is syntactically valid."""
    validator = HTMLValidator()
    try:
        validator.feed(html_code)
        return validator.is_valid()
    except Exception:
        return False


def extract_tailwind_classes(code: str) -> List[str]:
    """Extract Tailwind CSS classes from HTML/JSX code."""
    # Match className="..." or class="..." attributes
    class_patterns = [
        r'className=["\'](.*?)["\']',
        r'class=["\'](.*?)["\']',
    ]
    
    classes = []
    for pattern in class_patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            classes.extend(match.split())
    
    return classes


def extract_color_tokens(code: str) -> List[str]:
    """Extract color-related tokens (Tailwind colors and hex codes)."""
    colors = []
    
    # Tailwind color classes (e.g., bg-blue-500, text-red-600)
    tailwind_colors = re.findall(r'\b(?:bg|text|border)-(\w+)-\d+\b', code)
    colors.extend(tailwind_colors)
    
    # Hex colors
    hex_colors = re.findall(r'#[0-9a-fA-F]{3,6}\b', code)
    colors.extend(hex_colors)
    
    return colors


def compute_color_entropy(color_tokens: List[str]) -> float:
    """
    Compute entropy of color distribution.
    Low entropy = "blue/purple problem" (mode collapse)
    High entropy = diverse color usage
    """
    if not color_tokens:
        return 0.0
    
    counts = Counter(color_tokens)
    total = len(color_tokens)
    probs = [count / total for count in counts.values()]
    
    # Shannon entropy
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropy


def compute_tailwind_coverage(classes: List[str]) -> Dict[str, float]:
    """
    Measure diversity of Tailwind utility classes.
    
    Returns:
        - unique_classes: Number of unique classes used
        - category_coverage: Coverage of different Tailwind categories
    """
    if not classes:
        return {"unique_classes": 0, "layout_coverage": 0, "color_coverage": 0, "spacing_coverage": 0}
    
    unique = len(set(classes))
    
    # Count coverage of different Tailwind categories
    layout_classes = [c for c in classes if any(prefix in c for prefix in ['flex', 'grid', 'block', 'inline', 'hidden'])]
    color_classes = [c for c in classes if any(prefix in c for prefix in ['bg-', 'text-', 'border-'])]
    spacing_classes = [c for c in classes if any(prefix in c for prefix in ['p-', 'm-', 'px-', 'py-', 'mx-', 'my-'])]
    
    return {
        "unique_classes": unique,
        "layout_coverage": len(set(layout_classes)) / max(len(layout_classes), 1),
        "color_coverage": len(set(color_classes)) / max(len(color_classes), 1),
        "spacing_coverage": len(set(spacing_classes)) / max(len(spacing_classes), 1),
    }


def compute_critical_token_accuracy(pred_tokens: List[str], true_tokens: List[str]) -> float:
    """
    Compute accuracy on "critical" tokens: HTML tags, Tailwind classes, CSS properties.
    These are more important than generic tokens for UI code quality.
    """
    # Define critical token patterns
    critical_patterns = [
        r'^<[a-zA-Z]+>?$',  # HTML tags
        r'^[a-z]+-[a-z0-9-]+$',  # Tailwind classes (e.g., bg-blue-500)
        r'^#[0-9a-fA-F]+$',  # Hex colors
        r'^[a-z]+:$',  # CSS properties (e.g., color:)
    ]
    
    critical_correct = 0
    critical_total = 0
    
    for pred, true in zip(pred_tokens, true_tokens):
        is_critical = any(re.match(pattern, true) for pattern in critical_patterns)
        if is_critical:
            critical_total += 1
            if pred == true:
                critical_correct += 1
    
    if critical_total == 0:
        return 0.0
    
    return critical_correct / critical_total


class UICodeMetricsCallback(TrainerCallback):
    """
    Custom callback to log UI-specific metrics during training.
    
    Monitors:
    1. Perplexity (from loss)
    2. Syntax validity rate
    3. Tailwind coverage
    4. Color entropy (detect "blue/purple" problem)
    5. Sample generations
    """
    
    def __init__(self, tokenizer, val_samples: Optional[List[Dict]] = None, log_every: int = 1):
        """
        Args:
            tokenizer: Tokenizer for decoding
            val_samples: List of validation samples for generation
            log_every: Log metrics every N steps (default: 1 for frequent monitoring)
        """
        self.tokenizer = tokenizer
        self.val_samples = val_samples or []
        self.log_every = log_every
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
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
        model.eval()
        
        all_colors = []
        all_classes = []
        valid_syntax_count = 0
        sample_generations = []
        
        # Generate from validation samples
        num_samples = min(5, len(self.val_samples))
        for i, sample in enumerate(self.val_samples[:num_samples]):
            prompt = sample.get("prompt", "")
            ground_truth = sample.get("answer", "")
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract code (remove prompt)
            generated_code = generated[len(prompt):].strip()
            
            # Compute metrics
            is_valid = compute_syntax_validity(generated_code)
            valid_syntax_count += int(is_valid)
            
            colors = extract_color_tokens(generated_code)
            all_colors.extend(colors)
            
            classes = extract_tailwind_classes(generated_code)
            all_classes.extend(classes)
            
            # Save for W&B table
            sample_generations.append({
                "step": step,
                "sample": i,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "generated_code": generated_code[:500] + "..." if len(generated_code) > 500 else generated_code,
                "ground_truth": ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth,
                "syntax_valid": is_valid,
                "num_colors": len(colors),
            })
        
        # Aggregate metrics
        syntax_validity_rate = valid_syntax_count / num_samples if num_samples > 0 else 0
        color_entropy = compute_color_entropy(all_colors)
        tailwind_coverage = compute_tailwind_coverage(all_classes)
        
        # Log to W&B
        wandb.log({
            "eval/syntax_validity_rate": syntax_validity_rate,
            "eval/color_entropy": color_entropy,
            "eval/unique_tailwind_classes": tailwind_coverage["unique_classes"],
            "eval/layout_coverage": tailwind_coverage["layout_coverage"],
            "eval/color_coverage": tailwind_coverage["color_coverage"],
            "eval/spacing_coverage": tailwind_coverage["spacing_coverage"],
        }, step=step)
        
        # Log sample generations as table
        if sample_generations:
            wandb.log({
                "eval/sample_generations": wandb.Table(
                    columns=list(sample_generations[0].keys()),
                    data=[list(s.values()) for s in sample_generations]
                )
            }, step=step)
        
        model.train()


def add_ui_metrics_to_trainer(trainer, tokenizer, val_samples: Optional[List[Dict]] = None):
    """
    Add UI code metrics callback to an existing trainer.
    
    Usage:
        trainer = SFTTrainer(...)
        add_ui_metrics_to_trainer(trainer, tokenizer, val_samples)
    """
    callback = UICodeMetricsCallback(
        tokenizer=tokenizer,
        val_samples=val_samples,
        log_every=1  # Log every step for frequent monitoring
    )
    trainer.add_callback(callback)
    return trainer
