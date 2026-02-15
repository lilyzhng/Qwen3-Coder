# Real-World Qwen Quantization & Post-Training References

Here are real-world examples of how people have quantized Qwen models and used them for post-training. The most common patterns are:

1.  **GPTQ Quantization → LoRA Fine-tuning:** Quantize the base model to 4-bit GPTQ, then load the quantized model and train LoRA adapters on top.
2.  **Unsloth/bitsandbytes QLoRA:** Load the bf16 model directly with on-the-fly 4-bit quantization and train LoRA adapters (QLoRA).


## Reference Table

| # | Title | Link | Key Takeaway |
|---|---|---|---|
| 1 | **Official Qwen GPTQ Docs** | [Qwen Docs](https://qwen.readthedocs.io/en/latest/quantization/gptq.html) | Official guide for quantizing Qwen with AutoGPTQ. |
| 2 | **Kaitchup: Qwen1.5 Quant & FT** | [Substack](https://kaitchup.substack.com/p/fine-tuning-and-quantization-of-qwen15) | Full workflow: QLoRA FT → GPTQ/AWQ quantize. |
| 3 | **HF Gist: FT GPTQ Model** | [GitHub Gist](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) | Official HF example of loading pre-quantized GPTQ model for LoRA training. |
| 4 | **Unsloth: Run & Fine-tune Qwen3** | [Unsloth Blog](https://unsloth.ai/blog/qwen3) | Shows how to fine-tune Unsloth's pre-quantized `bnb-4bit` models. |
| 5 | **Unsloth Qwen3 Colab** | [Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb) | Free, runnable notebook for fine-tuning Unsloth's quantized Qwen3. |
| 6 | **Kaggle: FT Qwen-2.5-Coder-14B** | [Kaggle](https://www.kaggle.com/code/ksmooi/fine-tuning-qwen-2-5-coder-14b-llm-sft-peft) | Real-world example of fine-tuning Qwen Coder with Unsloth. |
| 7 | **Medium: FT GPTQ VLM** | [Medium](https://medium.com/@arunsreekuttan1996/fine-tuning-gptq-quantized-vision-language-models-with-lora-733d1e687ff5) | Shows fine-tuning a GPTQ-quantized Qwen-VL model with LoRA. |
| 8 | **Reddit: FT Qwen 2.5 Coder** | [Reddit](https://www.reddit.com/r/unsloth/comments/1jdi72l/i_finetuned_qwen_25_coder_on_a_single_repo_using/) | User reports 47% improvement in code completion after fine-tuning with Unsloth. |
| 9 | **Qwen Official Quantized Models** | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4) | Official GPTQ-quantized models from the Qwen team, ready for inference or fine-tuning. |
