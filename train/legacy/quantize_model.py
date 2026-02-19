"""
Pre-quantize Qwen3-Coder-Next-Base to 4-bit GPTQ and push to HuggingFace.

Based on:
- finetuning/Quantization Workflow Guide_ Qwen3-Coder-Next on Modal.md
- finetuning/Real-World Qwen Quantization & Post-Training References.md
- Official Qwen GPTQ docs: https://qwen.readthedocs.io/en/latest/quantization/gptq.html

Uses AutoGPTQ directly (as recommended by Qwen docs) with max_memory for
multi-GPU loading. Calibration uses our UIGEN training data.

Usage:
    modal run finetuning/quantize_model.py --hf-repo lilyzhng/Qwen3-Coder-Next-Base-GPTQ-4bit
"""

import modal

app = modal.App("uiux-quantize")

quantize_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Install torch first (auto-gptq needs it at build time)
    .pip_install("torch>=2.7.1")
    .pip_install(
        "accelerate",
        "gptqmodel",
        "datasets",
        "huggingface_hub",
        "hf-transfer",
        "optimum>=1.20.0",
    )
    # Transformers from source for qwen3_next architecture support
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .env({"HF_HOME": "/model_cache"})
)

model_cache_vol = modal.Volume.from_name("uiux-model-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("uiux-training-data", create_if_missing=True)


@app.function(
    image=quantize_image,
    gpu="A100-80GB:4",  # 4x A100 80GB = 320GB (bf16 model = 160GB + overhead)
    volumes={
        "/model_cache": model_cache_vol,
        "/training_data": data_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=5 * 60 * 60,  # 5 hours
)
def quantize_and_push(model_name: str, hf_repo: str):
    """Load model in bf16 across 4 GPUs, GPTQ-quantize to 4-bit, push to HF."""
    import json
    import logging
    import os

    import torch
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Print GPU info
    n_gpus = torch.cuda.device_count()
    total_mem = 0
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_mem += mem_gb
        print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
    print(f"  Total VRAM: {total_mem:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPTQ quantization config (following Qwen official docs)
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        damp_percent=0.01,
        desc_act=False,       # Faster inference, minimal quality impact
        sym=True,              # Symmetric quantization
        true_sequential=True,  # Better quality
    )

    # Load model across 4 GPUs using max_memory (recommended by Qwen docs)
    # Each GPU gets ~75GB allocation, leaving headroom for quantization overhead
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    print(f"\nLoading {model_name} in bf16 across {n_gpus} GPUs...")
    print(f"  max_memory: {max_memory}")
    model = GPTQModel.load(
        model_name,
        quantize_config,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
    )

    # Cache model files on volume
    model_cache_vol.commit()
    print("Model files cached to volume")

    # Prepare calibration data from our UIGEN training set
    # (Qwen docs recommend using training data for calibration)
    print("\nPreparing calibration data from UIGEN training set...")
    train_path = "/training_data/data/uigen_train.jsonl"
    max_len = 8192

    calibration_data = []
    with open(train_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            text = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            model_inputs = tokenizer([text])
            input_ids = torch.tensor(
                model_inputs.input_ids[0][:max_len], dtype=torch.int
            ).unsqueeze(0)
            calibration_data.append(
                dict(
                    input_ids=input_ids,
                    attention_mask=input_ids.ne(tokenizer.pad_token_id),
                )
            )

    print(f"  Calibration samples: {len(calibration_data)}")

    # Run GPTQ quantization
    print("\nQuantizing model to GPTQ 4-bit...")
    print("  This may take 1-3 hours for an 80B model...")
    model.quantize(calibration_data)

    # Save quantized model
    save_path = "/model_cache/quantized_model"
    print(f"\nSaving quantized model to {save_path}...")
    model.save(save_path)
    tokenizer.save_pretrained(save_path)

    # Check saved model size
    total_size = 0
    for root, dirs, files in os.walk(save_path):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"  Quantized model size: {total_size / 1e9:.1f} GB")

    # Push to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_repo and hf_token:
        print(f"\nPushing quantized model to HuggingFace: {hf_repo}...")
        # Load with transformers for push_to_hub compatibility
        from transformers import AutoModelForCausalLM
        push_model = AutoModelForCausalLM.from_pretrained(
            save_path,
            device_map="cpu",
        )
        push_model.push_to_hub(hf_repo, token=hf_token, private=True)
        tokenizer.push_to_hub(hf_repo, token=hf_token, private=True)
        print(f"  Pushed to https://huggingface.co/{hf_repo}")
    else:
        print("Skipping HuggingFace push (no repo or token)")

    model_cache_vol.commit()
    print("\nDone! Quantized model is ready for single-GPU fine-tuning.")
    return hf_repo


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-Coder-Next-Base",
    hf_repo: str = "",
):
    if not hf_repo:
        print("ERROR: --hf-repo is required.")
        print("Usage: modal run finetuning/quantize_model.py --hf-repo lilyzhng/Qwen3-Coder-Next-Base-GPTQ-4bit")
        return

    print(f"GPTQ Quantization Job (following Qwen official docs)")
    print(f"  Source model: {model_name}")
    print(f"  Target repo:  {hf_repo}")
    print(f"  GPUs: 4x A100 80GB")
    print(f"  Method: GPTQ 4-bit, group_size=128, desc_act=False, sym=True")
    print(f"  Calibration: UIGEN training data")
    print()

    result = quantize_and_push.remote(model_name, hf_repo)
    print(f"\nQuantized model pushed to: https://huggingface.co/{result}")
    print(f"\nTo fine-tune on 1 GPU:")
    print(f"  modal run finetuning/modal_train.py --model-name {result}")
