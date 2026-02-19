# PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization

<div align="center">

<img alt="PlotCraft Main Fig" src="plotcraft_abstract_fig.jpg"/>

</div>
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2511.00010-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2511.00010)
</div>

**PlotCraft** is a rigorous benchmark designed to evaluate the advanced data visualization capabilities of LLMs. It presents ~1k challenging tasks to assess how well models can generate and refine complex plots from natural language instructions.

**Key Features**:

* **Comprehensive Scope**: Includes 982 tasks covering 48 chart types across 8 major domains (e.g., Finance, Health, Research).
* **Dual Evaluation Modes**: The first benchmark to systematically test both:
Single-Turn Generation: From scratch, based on an initial request.
* **Multi-Turn Refinement**: Iteratively debugging and enhancing existing code.
* **Focus on Complexity**: Tasks are designed with compositional complexity, requiring multi-panel layouts and combined chart types to test a model's spatial and logical reasoning.
* **Realistic Workflow**: Built from scratch using real-world datasets and zero-reference instructions (text-only), simulating a practical data analyst workflow.


## Installation
```bash
conda create -n plotcraftbench python=3.13
conda activate plotcraftbench
pip install -r requirements.txt
```

## Prepare Data

```bash
zip -F data.zip --out complete_data.zip
unzip complete_data.zip -d data
```

### Method 1: Automated Download via Kaggle API

1. **Obtain Kaggle API credentials**: Navigate to your Kaggle account settings (Account tab) and select 'Create New Token'. This downloads `kaggle.json` containing your API credentials.

2. **Configure credentials**: Move `kaggle.json` to the appropriate location:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`

3. **Install Kaggle CLI**:
```bash
pip install kaggle
```

4. **Download datasets**:
```bash
python download_datasets.py data/
```

### Method 2: Manual Download

Alternatively, download datasets manually using the URLs specified in `download_url.json` files located in each subdirectory (e.g., `data/<dataset-name>/download_url.json`). After downloading, place all CSV and XLSX files in the root of their respective subdirectories.

## Quick Start

### Configure API Credentials

Edit `run_single_turn.sh` or `run_multi_turn.sh` to set your OpenAI-compatible API endpoint and keys:

```bash
API_KEY=""          # Evaluation API key
API_KEY_GEN=""      # Generation API key
```

### Run Evaluation

**Single-Turn Mode** (generate plots from scratch):
```bash
bash run_single_turn.sh <model_name> <api_base_url>
```

**Multi-Turn Mode** (iterative refinement):
```bash
bash run_multi_turn.sh <model_name> <api_base_url>
```


Results will be saved to `results_single_turn/` (single-turn) or `results_multi_turn/` (multi-turn).

