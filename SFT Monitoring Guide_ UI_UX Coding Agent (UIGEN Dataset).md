# SFT Monitoring Guide: UI/UX Coding Agent (UIGEN Dataset)

During the **Supervised Fine-Tuning (SFT)** phase, your goal is to maximize the model's "imitation accuracy" of the high-quality UI code and reasoning traces in the UIGEN dataset. Unlike RLHF/DPO, which focuses on preferences, SFT is about **structural correctness** and **knowledge acquisition**.

## 1. Primary SFT Losses & Matrices

| Metric | Importance | Interpretation for UI/UX |
| :--- | :--- | :--- |
| **Cross-Entropy Loss** | **Critical** | Measures the "surprise" of the model compared to the UIGEN reference. A steady, smooth decrease is essential. |
| **Perplexity (PPL)** | **High** | Indicates how well the model predicts the next token in a UI component. High PPL usually means broken syntax or messy CSS. |
| **Token Accuracy** | **High** | The % of tokens the model predicts correctly vs. the ground truth. For UI code, aim for **70-85%** accuracy. |
| **Critical Token Accuracy** | **Very High** | Focus specifically on "high-stakes" tokens: HTML tags, Tailwind class names, and CSS property names. |

## 2. UIGEN-Specific Monitoring (Reasoning + Code)

Since the UIGEN dataset (like T1.1) uses **Reasoning Traces (Chain-of-Thought)**, you must monitor two distinct parts of the generation:

### A. Reasoning Trace Quality (The "Think" block)
*   **Trace Length Stability:** If the reasoning traces start getting shorter or incoherent, the model is "skipping steps," which leads to poor UI layout decisions.
*   **Formatting Adherence:** Ensure the model correctly uses the `<|im_start|>think` and `<|im_start|>answer` tokens. Loss spikes often occur when the model fails to transition between reasoning and code.

### B. UI Code Integrity (The "Answer" block)
*   **Syntax Validity Rate:** Automatically check if the generated HTML/JSX is valid. SFT should quickly push this toward 100%.
*   **Tailwind/CSS Coverage:** Monitor the diversity of CSS classes. If the model only outputs `bg-blue-500` (the "blue/purple" problem), your SFT loss may be plateauing on "safe" but generic tokens.

## 3. The "Blue/Purple" Regression Check

The "blue/purple website" problem is a form of **Mode Collapse** in SFT. Even if your loss is decreasing, the model might be defaulting to the most frequent (and boring) patterns in the base model.

*   **Entropy of Color Tokens:** Monitor the diversity of hex codes or Tailwind color classes in your validation samples. A sharp drop in color entropy means the model is reverting to "generic agent" mode.
*   **Visual Regression (Manual):** Every 50 steps, render a few validation samples. If the designs look identical despite different prompts, your learning rate might be too low, or you are overfitting to a small subset of the data.

## 4. Early Warning Signs in SFT

*   **Loss Spikes at Sequence Ends:** If loss spikes toward the end of long UI components, the model is struggling with **long-range dependencies** (e.g., closing a deeply nested `<div>`). Consider increasing the `max_seq_length`.
*   **Validation Loss Divergence:** If training loss keeps dropping but validation loss starts rising, you are **overfitting**. Since UIGEN-T1.1 is a small dataset (~700 samples), this can happen very quickly (often in 1-2 epochs).
*   **Instruction Forgetting:** If the model starts ignoring the "User" prompt and just outputs a generic dashboard from the dataset, your "Prompt Masking" during SFT might be incorrect. Ensure you are only calculating loss on the *Assistant's* response, not the *User's* prompt.

## 5. Recommended Evaluation Checklist

1.  **Step 100:** Check if the `<|im_start|>think` format is being followed.
2.  **Step 200:** Verify that Tailwind classes are actually being used instead of raw CSS.
3.  **Step 500:** Perform a "Visual Audit"—do the layouts match the complexity of the UIGEN samples?
4.  **End of Epoch 1:** Compare against the base Qwen2.5-Coder. The SFT model should be significantly more "opinionated" about design layout.

## References

[1] ArXiv: Fait: Fault-Aware Fine-Tuning for Better Code Generation. [https://arxiv.org/html/2503.16913v1](https://arxiv.org/html/2503.16913v1)
[2] Tesslate: UIGEN-T1.1-Qwen-14B Model Card. [https://huggingface.co/Tesslate/UIGEN-T1.1-Qwen-14B](https://huggingface.co/Tesslate/UIGEN-T1.1-Qwen-14B)
[3] Hugging Face: Supervised Fine-Tuning (SFT) Course. [https://huggingface.co/learn/llm-course/en/chapter11/3](https://huggingface.co/learn/llm-course/en/chapter11/3)
[4] iMerit: Supervised Fine-Tuning for Text-to-Code Models. [https://imerit.net/resources/blog/supervised-fine-tuning-for-text-to-code-models-building-smarter-ai-developers/](https://imerit.net/resources/blog/supervised-fine-tuning-for-text-to-code-models-building-smarter-ai-developers/)
[5] ArXiv: Benchmarks and Metrics for Evaluations of Code Generation. [https://arxiv.org/html/2406.12655v1](https://arxiv.org/html/2406.12655v1)
