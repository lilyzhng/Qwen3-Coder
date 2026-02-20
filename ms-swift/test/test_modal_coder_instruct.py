"""
Tests for modal_coder_instruct.py dataset loading pipeline.

Driven by symptom: only 50 training samples observed instead of expected 645.
Each test isolates one step in the pipeline so the failure tells us exactly
where the count drops.

Run locally (HF-only tests, no ms-swift import needed):
    cd /Users/lilyzhang/Desktop/Archive/ms-swift
    python -m pytest Qwen3-Coder/ms-swift/test/test_modal_coder_instruct.py -v -k "hf_"

Run on Modal (full suite, ms-swift installed):
    modal run Qwen3-Coder/ms-swift/test/test_modal_coder_instruct.py
"""

import modal

# Re-use the same image and secrets as the training script
_image = (
    modal.Image.from_registry('nvidia/cuda:12.8.0-devel-ubuntu22.04', add_python='3.11')
    .apt_install('git', 'build-essential')
    .pip_install(
        'ms-swift @ git+https://github.com/modelscope/ms-swift.git',
        'transformers>=4.57,<4.58',
        'trl<0.25',
        'datasets',
        'pytest',
    )
    .pip_install('hf_transfer')
    .env({'USE_HF': '1', 'HF_HUB_ENABLE_HF_TRANSFER': '1'})
)

app = modal.App('test-modal-coder-instruct')


@app.function(image=_image, timeout=300)
def run_tests_remote():
    """Run the full test suite inside the ms-swift Modal environment."""
    import subprocess
    import textwrap

    # Write the test file inline (can't mount local files in a function call)
    test_code = textwrap.dedent('''
        import sys, os
        import pytest

        HF_DATASET = "lilyzhng/UIGEN-T1.1-split"
        EXPECTED_TRAIN = 645

        def test_hf_train_split_size():
            from datasets import load_dataset
            ds = load_dataset(HF_DATASET, split="train")
            assert len(ds) == EXPECTED_TRAIN, f"Got {len(ds)}, expected {EXPECTED_TRAIN}"

        def test_dataset_syntax_no_sample_limit():
            from swift.dataset.dataset_syntax import DatasetSyntax
            syntax = DatasetSyntax.parse(HF_DATASET)
            assert syntax.dataset_sample is None, f"Unexpected sample limit: {syntax.dataset_sample}"

        def test_dataset_meta_uses_train_split():
            from swift.dataset.dataset_syntax import DatasetSyntax
            syntax = DatasetSyntax.parse(HF_DATASET)
            meta = syntax.get_dataset_meta(use_hf=True)
            assert meta.split == ["train"], f"Expected [train], got {meta.split}"

        def test_auto_preprocessor_keeps_all_rows():
            from datasets import load_dataset
            from swift.dataset.preprocessor.core import AutoPreprocessor
            ds = load_dataset(HF_DATASET, split="train")
            result = AutoPreprocessor()(ds, num_proc=1, load_from_cache_file=False, strict=False)
            assert len(result) == EXPECTED_TRAIN, (
                f"AutoPreprocessor dropped rows: {EXPECTED_TRAIN} -> {len(result)}"
            )

        def test_messages_preprocessor_single_row():
            from datasets import load_dataset
            from swift.dataset.preprocessor.core import MessagesPreprocessor
            ds = load_dataset(HF_DATASET, split="train")
            row = dict(ds[0])
            result = MessagesPreprocessor().preprocess(row)
            assert result is not None, "MessagesPreprocessor returned None for row 0"

        def test_swift_load_dataset_count():
            from swift.dataset import load_dataset as swift_load_dataset
            train_ds, val_ds = swift_load_dataset(
                [HF_DATASET], use_hf=True, split_dataset_ratio=0.0,
                num_proc=1, load_from_cache_file=False,
            )
            assert len(train_ds) == EXPECTED_TRAIN, (
                f"swift load_dataset returned {len(train_ds)}, expected {EXPECTED_TRAIN}"
            )

        def test_swift_load_dataset_cache_consistent():
            """Cached and uncached loads must agree — catches stale cache bugs."""
            from swift.dataset import load_dataset as swift_load_dataset
            train_no_cache, _ = swift_load_dataset(
                [HF_DATASET], use_hf=True, split_dataset_ratio=0.0,
                num_proc=1, load_from_cache_file=False,
            )
            train_with_cache, _ = swift_load_dataset(
                [HF_DATASET], use_hf=True, split_dataset_ratio=0.0,
                num_proc=1, load_from_cache_file=True,
            )
            assert len(train_no_cache) == len(train_with_cache), (
                f"Cache mismatch! no_cache={len(train_no_cache)}, "
                f"with_cache={len(train_with_cache)}. Stale cache detected."
            )

        if __name__ == "__main__":
            pytest.main([__file__, "-v"])
    ''')

    test_path = '/tmp/test_dataset.py'
    with open(test_path, 'w') as f:
        f.write(test_code)

    result = subprocess.run(
        ['python', '-m', 'pytest', test_path, '-v', '--tb=short'],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr)
    return result.returncode


@app.local_entrypoint()
def main():
    exit_code = run_tests_remote.remote()
    if exit_code != 0:
        raise SystemExit(f'Tests failed (exit code {exit_code})')

import sys
import os

import pytest

HF_DATASET = 'lilyzhng/UIGEN-T1.1-split'
EXPECTED_TRAIN = 645
EXPECTED_VAL = 80
EXPECTED_TEST = 80


# ── Step 1: raw HuggingFace dataset ──────────────────────────────────────────

def test_hf_train_split_size():
    """HF dataset has 645 train samples — verifies upload was correct."""
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split='train')
    assert len(ds) == EXPECTED_TRAIN, (
        f'HF train split has {len(ds)} samples, expected {EXPECTED_TRAIN}. '
        'Check that prepare_uigen_data.py --upload ran correctly.'
    )


def test_hf_splits_exist():
    """All three splits (train/validation/test) exist with correct sizes."""
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET)
    assert 'train' in ds, 'Missing train split'
    assert 'validation' in ds, 'Missing validation split'
    assert 'test' in ds, 'Missing test split'
    assert len(ds['train']) == EXPECTED_TRAIN
    assert len(ds['validation']) == EXPECTED_VAL
    assert len(ds['test']) == EXPECTED_TEST


def test_hf_train_messages_format():
    """Train samples have non-empty messages in system/user/assistant format."""
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split='train')
    bad = []
    for i, row in enumerate(ds):
        msgs = row.get('messages', [])
        if not msgs:
            bad.append((i, 'empty messages'))
            continue
        roles = [m['role'] for m in msgs]
        if 'user' not in roles or 'assistant' not in roles:
            bad.append((i, f'missing user/assistant, got roles: {roles}'))
    assert not bad, f'{len(bad)} malformed rows (first 5): {bad[:5]}'


# ── Step 2: ms-swift DatasetSyntax parsing ───────────────────────────────────

def test_dataset_syntax_parse():
    """ms-swift parses the dataset string without adding a sample limit."""
    sys.path.insert(0, os.path.abspath('.'))
    from swift.dataset.dataset_syntax import DatasetSyntax

    syntax = DatasetSyntax.parse(HF_DATASET)
    assert syntax.dataset == HF_DATASET
    assert syntax.dataset_sample is None, (
        f'DatasetSyntax added an unexpected sample limit: {syntax.dataset_sample}'
    )
    assert syntax.dataset_type == 'repo'


def test_dataset_meta_split_default():
    """The DatasetMeta used for our unregistered dataset defaults to split=['train']."""
    sys.path.insert(0, os.path.abspath('.'))
    from swift.dataset.dataset_syntax import DatasetSyntax

    syntax = DatasetSyntax.parse(HF_DATASET)
    meta = syntax.get_dataset_meta(use_hf=True)
    assert meta.split == ['train'], (
        f'Expected split=[\'train\'], got {meta.split}'
    )


# ── Step 3: ms-swift AutoPreprocessor ────────────────────────────────────────

def test_auto_preprocessor_output_size():
    """AutoPreprocessor should keep all 645 rows — if it drops any, this fails."""
    sys.path.insert(0, os.path.abspath('.'))
    from datasets import load_dataset
    from swift.dataset.preprocessor.core import AutoPreprocessor

    ds = load_dataset(HF_DATASET, split='train')
    assert len(ds) == EXPECTED_TRAIN, 'Precondition: raw dataset must be correct'

    result = AutoPreprocessor()(ds, num_proc=1, load_from_cache_file=False, strict=False)
    assert len(result) == EXPECTED_TRAIN, (
        f'AutoPreprocessor reduced dataset from {EXPECTED_TRAIN} to {len(result)}. '
        'Some rows are being silently dropped during preprocessing.'
    )


def test_messages_preprocessor_on_single_row():
    """A single train row survives MessagesPreprocessor without modification."""
    sys.path.insert(0, os.path.abspath('.'))
    from datasets import load_dataset
    from swift.dataset.preprocessor.core import MessagesPreprocessor

    ds = load_dataset(HF_DATASET, split='train')
    row = dict(ds[0])  # first row as plain dict

    preprocessor = MessagesPreprocessor()
    result = preprocessor.preprocess(row)
    assert result is not None, (
        f'MessagesPreprocessor.preprocess() returned None for row 0. '
        f'Messages: {row.get("messages", [])[:1]}'
    )
    assert len(result['messages']) > 0


# ── Step 4: ms-swift load_dataset end-to-end ─────────────────────────────────

def test_swift_load_dataset_count():
    """ms-swift's load_dataset() must return 645 training samples, not 50."""
    sys.path.insert(0, os.path.abspath('.'))
    from swift.dataset import load_dataset as swift_load_dataset

    train_ds, val_ds = swift_load_dataset(
        [HF_DATASET],
        use_hf=True,
        split_dataset_ratio=0.0,  # no internal val split
        num_proc=1,
        load_from_cache_file=False,
    )
    assert train_ds is not None
    assert len(train_ds) == EXPECTED_TRAIN, (
        f'swift load_dataset returned {len(train_ds)} samples, expected {EXPECTED_TRAIN}. '
        f'val_ds size: {len(val_ds) if val_ds else None}'
    )


def test_swift_load_dataset_no_internal_split():
    """With split_dataset_ratio=0 and no val_dataset, ms-swift must not split train."""
    sys.path.insert(0, os.path.abspath('.'))
    from swift.dataset import load_dataset as swift_load_dataset

    train_ds, val_ds = swift_load_dataset(
        [HF_DATASET],
        use_hf=True,
        split_dataset_ratio=0.0,
        num_proc=1,
        load_from_cache_file=False,
    )
    assert val_ds is None or len(val_ds) == 0, (
        f'ms-swift created an unexpected val split of {len(val_ds)} samples '
        'even though split_dataset_ratio=0.'
    )
