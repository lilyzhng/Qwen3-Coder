# CLAUDE.md — Qwen3-Coder

This repo contains eval scripts, training configs, and tooling for fine-tuning Qwen3-Coder-Next (80B MoE, 3B active) on UI/UX code generation.

## Project Structure

```
eval/                   # Eval pipeline
  modal_eval_instruct.py  # Main eval script (runs on Modal H200)
  test_judge.py           # Local judge test harness (pytest)
  modal_eval_moe.py       # MoE-specific eval
training_data.json      # Bug fix conversations (post-training data, source of truth)
PROGRESS.md             # Symlink — all bug fixes + eval results
journal/                # Symlink — daily journal (YYYYMMDD.md)
ms-swift/               # Nested ms-swift framework (submodule)
unsloth/                # Unsloth MoE training scripts
finetuning/             # LoRA fine-tuning configs
```

## Eval Pipeline

### Running eval on Modal
```bash
cd eval
# Full eval (costs ~$10+ on H200):
modal run modal_eval_instruct.py --limit 5 --lora-model <hf_repo_id>

# Download results only:
modal run modal_eval_instruct.py --download-only                    # list runs
modal run modal_eval_instruct.py --download-only --run-name <name>  # download specific run
```

Results download to `wandb/eval_results/comparison-...` with: `{id}_base_raw.txt`, `{id}_base.html`, `{id}_gt.html`, `screenshots/{id}_*.png`, `{id}_*_judgment.json`

### Running tests locally
```bash
# Unit tests (no API key needed):
python -m pytest eval/test_judge.py -v -k "parse or extract"

# Integration tests (needs OPENROUTER_API_KEY):
python -m pytest eval/test_judge.py -v -k "judge"
```

## Automated Workflows

These are MANDATORY — do them automatically, never wait for the user to ask.

### After every bug fix commit+push:
1. **PROGRESS.md** — Add entry with: bug ID (e.g. `ms-swift-28`), timestamp (from `date` command), affected files, commit hash, error, root cause, fix, test results
2. **journal/YYYYMMDD.md** — Add concise summary under `#### Build + Learn` with bug ID, what broke, what fixed it, commit hash
3. **training_data.json** — If the debugging conversation is worth training on, append an entry (see format below). Skip for trivial fixes.

### After changing eval code (judge prompt, rubric, parsing, extract_code, etc.):
1. **Download latest eval results** from Modal: `modal run modal_eval_instruct.py --download-only --run-name <latest>`
2. **Run local judge test** against the downloaded fixtures using the updated code — compare old vs new judgment
3. **Show before/after** to the user (old score/failure_modes vs new)
4. The point: validate eval changes locally before launching expensive Modal runs

## Bug Fix Workflow

When a bug is found and fixed:
1. Fix the code
2. Commit + push (on a feature branch, never main)
3. Auto-document in PROGRESS.md and journal (see above)
4. Append to `training_data.json` if it's a non-trivial debugging conversation

## Training Data Format

`training_data.json` is the single source of truth for bug fix conversations. Each entry:

```json
{
  "bug_id": "ms-swift-28",
  "title": "Visible code artifacts in rendered screenshots",
  "timestamp": "2026-02-25 22:02 PST",
  "commit": "d9d122c",
  "path": "Qwen3-Coder",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Daily Journal

- Journal lives at `./journal/` (symlink to `/Users/lilyzhang/Documents/lilyzhng/2026/`)
- Filename format: `YYYYMMDD.md` (no hyphens), e.g. `journal/20260225.md`
- At conversation start, read today's journal to load TODOs and context
- Key sections: `#### Build + Learn` (tech TODOs), `#### Ship` (artifacts), `#### Work` (work context)

## Code Standards

- Python 3.12+, type hints where useful
- Use `pytest` for tests
- Timestamps: always use `date` command, never write manually
- Line length: 120 characters
