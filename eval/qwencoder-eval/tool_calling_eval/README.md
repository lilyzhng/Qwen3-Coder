# Evaluation Guide for Tau-Bench and BFCL-v3

This guide provides instructions for evaluating the model on the **Tau-Bench** and **Berkeley Function-Calling Leaderboard (BFCL-v3)** benchmarks.

## 1\. Prerequisites: API Setup

Our evaluation scripts use **LiteLLM** to interface with various model endpoints. Before you begin, ensure you have a `litellm` server running that provides an OpenAI-compatible API for the model you want to evaluate.

For detailed instructions, please refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/).

Once your API endpoint is active, export the following environment variables in your terminal:

```bash
export OPENAI_API_BASE="YOUR_API_BASE_URL"
export OPENAI_API_KEY="YOUR_API_KEY"
```

-----

## 2\. Tau-Bench Evaluation

### Step 1: Setup Environment

Navigate to the `tau-bench` directory, create a virtual environment, and install the required packages.

```bash
cd tau-bench
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Step 2: Run Evaluation

Execute the provided scripts to run the evaluation on the **retail** and **airline** domains.

```bash
# Evaluate the retail domain
bash retail-qwen3-coder.bash

# Evaluate the airline domain
bash airline-qwen3-coder.bash
```

-----

## 3\. BFCL-v3 Evaluation

The Berkeley Function-Calling Leaderboard (BFCL) assesses the model's function-calling capabilities.

### Step 1: Setup Environment

Navigate to the `berkeley-function-call-leaderboard` directory, create a virtual environment, and install the dependencies.

```bash
cd berkeley-function-call-leaderboard
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Step 2: Run Evaluation

First, generate the model's responses. Then, run the evaluation script to score the results.

```bash
# Generate model responses for Qwen3-Coder
bfcl generate --model Qwen3-Coder-480B-A35B-Instruct --num-threads 16

# Evaluate the results
bfcl evaluate --model Qwen3-Coder-480B-A35B-Instruct
```

-----

## Additional Resources

For more details on the benchmarks, please visit their official repositories:

  * **Tau-Bench:** [https://github.com/sierra-research/tau-bench](https://github.com/sierra-research/tau-bench)
  * **BFCL-v3:** [https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)