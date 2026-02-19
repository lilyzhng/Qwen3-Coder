# export OPENAI_API_BASE=YOUR_API_BASE
# export OPENAI_API_KEY=YOUR_API_KEY

python run.py --num-trials 1 --agent-strategy tool-calling --env airline --model Qwen3-Coder --model-provider openai --user-model gpt-4o-2024-11-20 --user-model-provider openai --user-strategy llm --max-concurrency 20 --log-dir airline --temperature 0.0
