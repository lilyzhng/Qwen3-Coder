# Lessons Learned — Modal + ms-swift Training

## Bug 1: `git` not installed in container image

- **Bug:** `modal.Image.debian_slim()` doesn't include `git`. Installing `ms-swift @ git+https://...` via pip fails with `Cannot find command 'git'`.
- **Cost:** Failed image build, wasted iteration time.
- **Solution:** Add `.apt_install('git')` before `.pip_install(...)`.

## Bug 2: `hf-transfer` installed but not enabled

- **Bug:** `hf-transfer` was listed as a pip dependency but the env var `HF_HUB_ENABLE_HF_TRANSFER=1` was never set, so downloads used the slow default Python downloader.
- **Cost:** Model download took 1+ hours instead of ~10-15 minutes. GPU time wasted while downloading.
- **Solution:** Add `'HF_HUB_ENABLE_HF_TRANSFER': '1'` to the `.env({...})` block.

## Bug 3: Model downloaded at runtime on expensive GPU

- **Bug:** The 160GB model was downloaded inside the training function, which runs on an H100 (~$4/hr). Every minute spent downloading is billed at GPU rates.
- **Cost:** ~$2-4 wasted on GPU time just for downloading, per run.
- **Solution:** Move the download to the image build step using `.run_commands('huggingface-cli download Qwen/Qwen3-Coder-Next-Base --max-workers 10')`. Image builds run on CPU (cheaper) and are cached across runs.

## Bug 4: Slow sequential shard downloads

- **Bug:** `huggingface-cli download` defaults to limited concurrency. With 40 shards at ~3.7GB each, sequential downloads are slow.
- **Cost:** Extended download time (1+ hours for 160GB).
- **Solution:** Use `--max-workers 10` to download 10 shards in parallel.

## Bug 5: Missing `flash-linear-attention` and `causal-conv1d`

- **Bug:** Qwen3-Coder-Next-Base uses a hybrid architecture (Mamba-like layers) requiring `flash-linear-attention` and `causal-conv1d`. Without them, the model falls back to a slow torch implementation that uses more memory, causing OOM during weight conversion on H100 (80GB).
- **Cost:** OOM crash after spending ~$10+ on GPU time and model download. Complete run failure.
- **Solution:** Add `'causal-conv1d'` and `'flash-linear-attention'` to `.pip_install(...)`. If OOM persists, use H200 (141GB) instead of H100.

## Bug 6: `causal-conv1d` fails to build — missing `nvcc` in container

- **Bug:** `modal.Image.debian_slim()` doesn't include CUDA development tools. `causal-conv1d` is a CUDA extension that must compile from source, requiring `nvcc`. Without it, the build fails with `NameError: name 'bare_metal_version' is not defined` (because the setup.py tries to detect the CUDA toolkit version via `nvcc` and crashes when it's missing).
- **Cost:** Failed image build, wasted iteration time.
- **Solution:** Switch from `modal.Image.debian_slim(python_version='3.11')` to `modal.Image.from_registry('nvidia/cuda:12.8.0-devel-ubuntu22.04', add_python='3.11')`. The `devel` variant of NVIDIA's CUDA images includes `nvcc` and all headers needed to compile CUDA extensions.

## Bug 7: `causal-conv1d` fails to build — no pre-built wheel for torch 2.10

- **Bug:** `causal-conv1d` v1.6.0 only ships pre-built wheels up to torch 2.5. With torch 2.10, pip falls back to building from source. But pip's default build isolation creates a clean venv **without torch**, so the setup.py fails with `NameError: name 'bare_metal_version' is not defined` (or `ModuleNotFoundError: No module named 'torch'`) because it can't import torch to detect the CUDA version.
- **Cost:** Failed image build even after fixing Bug 6 (nvcc was present but torch wasn't visible during build).
- **Solution:** Install `causal-conv1d` separately with `--no-build-isolation` so it can find the already-installed torch: `.run_commands('pip install causal-conv1d --no-build-isolation')`. This must come **after** the main `.pip_install(...)` step that installs torch.


## Bug 8: `causal-conv1d` fails to build — missing C++ compiler in CUDA devel image

- **Bug:** The `nvidia/cuda:12.8.0-devel-ubuntu22.04` image includes CUDA development tools (`nvcc`) but **does not include a C++ compiler** (`g++` or `clang`). When pip tries to build `causal-conv1d` from source, it fails with `error: command 'clang' failed: No such file or directory`.
- **Cost:** Failed image build. Even though CUDA tools are present, the C++ extension can't compile without a C++ compiler.
- **Solution:** Install `build-essential` (which includes `g++`, `make`, and other build tools) in the image: `.apt_install('git', 'build-essential')`. This must come before the pip install steps that need to compile C++ extensions
