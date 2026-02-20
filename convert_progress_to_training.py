"""Convert PROGRESS.md bug fix sections to training_data.json format.

Parses every '#### ms-swift-N' / '#### unsloth-N' section in PROGRESS.md and
converts it to a training entry. New entries are appended; existing bug_ids are
skipped unless --force is passed.

Usage:
    # Preview what would be added (no writes)
    python Qwen3-Coder/convert_progress_to_training.py --dry-run

    # Append new entries to training_data.json
    python Qwen3-Coder/convert_progress_to_training.py

    # Overwrite all entries (re-parse everything)
    python Qwen3-Coder/convert_progress_to_training.py --force

    # Custom paths
    python Qwen3-Coder/convert_progress_to_training.py \\
        --progress /path/to/PROGRESS.md \\
        --output Qwen3-Coder/training_data.json
"""

import argparse
import json
import re
from pathlib import Path

PROGRESS_PATH = Path.home() / 'Documents/lilyzhng/Learn/RL_PostTrain/PROGRESS.md'
DEFAULT_OUTPUT = Path('Qwen3-Coder/training_data.json')

DEFAULT_SYSTEM = (
    'You are an ML infrastructure engineer. '
    'Project: fine-tuning Qwen3-Coder-Next (80B MoE, 3B active) on Modal '
    'using ms-swift with QLoRA (BNB 4-bit NF4). Hardware: H200/B200 GPU.'
)

# Matches: #### ms-swift-26: Some title
# or:      #### unsloth-5-fix: Some title
BUG_HEADING_RE = re.compile(
    r'^(#{2,6})\s+((ms-swift|unsloth)-([\w.-]+)):\s+(.+)$',
    re.MULTILINE,
)

# New format: **Date**: 2026-02-20 08:48 PST
NEW_DATE_RE = re.compile(r'^\*\*Date\*\*:\s*(.+)$', re.MULTILINE)
# Old format: 2026-02-17 19:30 PST — commit `12ba29f`
OLD_DATE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+\w+)', re.MULTILINE)

# New format: **Commit**: `c1d6406` ([view](...))
NEW_COMMIT_RE = re.compile(r'\*\*Commit\*\*:\s*`([0-9a-f]+)`')
# Old format: — commit `12ba29f`
OLD_COMMIT_RE = re.compile(r'—\s*commit\s*`([0-9a-f]+)`')


def extract_timestamp(body: str) -> str:
    m = NEW_DATE_RE.search(body) or OLD_DATE_RE.search(body)
    return m.group(1).strip() if m else ''


def extract_commit(body: str) -> str | None:
    m = NEW_COMMIT_RE.search(body) or OLD_COMMIT_RE.search(body)
    return m.group(1) if m else None


def extract_field(body: str, *field_names: str) -> str:
    """Extract content after **FieldName**: up to the next **Bold**: marker or end.

    Accepts multiple field_names to try in order (handles format variations like
    'Problem' vs 'Error', 'Root cause' vs 'Root Cause').
    """
    for name in field_names:
        # Allow optional trailing text in parens: **Wrong paths tried** (if any):
        pattern = re.compile(
            rf'\*\*{re.escape(name)}\*\*[^:]*:\s*(.*?)(?=\n\*\*[A-Z]|\Z)',
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(body)
        if m:
            return m.group(1).strip()
    return ''


def infer_path(bug_id: str) -> str:
    if bug_id.startswith('ms-swift'):
        return 'ms-swift'
    if bug_id.startswith('unsloth'):
        return 'unsloth'
    return bug_id.rsplit('-', 1)[0]


def section_to_entry(bug_id: str, title: str, body: str) -> dict:
    """Convert a PROGRESS.md section to a training_data.json entry."""
    timestamp = extract_timestamp(body)
    commit = extract_commit(body)
    path = infer_path(bug_id)

    # Try new-format fields first, fall back to old-format equivalents
    problem = extract_field(body, 'Problem', 'Error')
    root_cause = extract_field(body, 'Root cause', 'Root Cause', 'Cause')
    fix = extract_field(body, 'Fix')
    wrong_paths = extract_field(body, 'Wrong paths tried', 'Wrong path')
    side_note = extract_field(body, 'Side note', 'Note')

    # User message: the problem/error statement
    user_content = problem if problem else f'Bug: {title}'

    # Assistant message: root cause → wrong paths → fix
    assistant_parts = []
    if root_cause:
        assistant_parts.append(f'**Root cause**: {root_cause}')
    if wrong_paths:
        assistant_parts.append(f'**Wrong paths tried**: {wrong_paths}')
    if fix:
        assistant_parts.append(f'**Fix**: {fix}')
    if side_note:
        assistant_parts.append(f'**Note**: {side_note}')

    # Fallback for old-format sections that don't use structured fields:
    # use the whole body as the assistant turn (it reads as a complete explanation)
    if not assistant_parts:
        # Strip the metadata line (timestamp/commit) from the top of older sections
        body_clean = OLD_DATE_RE.sub('', body).strip()
        assistant_parts.append(body_clean)

    return {
        'bug_id': bug_id,
        'title': title,
        'timestamp': timestamp,
        'commit': commit,
        'path': path,
        'messages': [
            {'role': 'system', 'content': DEFAULT_SYSTEM},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': '\n\n'.join(assistant_parts)},
        ],
    }


def split_sections(md_text: str) -> list[tuple[str, str, str]]:
    """Return (bug_id, title, body) for every bug section in the markdown."""
    matches = list(BUG_HEADING_RE.finditer(md_text))
    results = []

    for i, m in enumerate(matches):
        heading_depth = len(m.group(1))
        bug_id = m.group(2)
        title = m.group(5).strip()

        start = m.end()
        end = len(md_text)

        # Body ends at the next heading of equal or higher level (fewer #s)
        for j in range(i + 1, len(matches)):
            next_depth = len(matches[j].group(1))
            if next_depth <= heading_depth:
                end = matches[j].start()
                break

        body = md_text[start:end].strip()
        results.append((bug_id, title, body))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert PROGRESS.md bug sections to training_data.json'
    )
    parser.add_argument(
        '--progress',
        default=str(PROGRESS_PATH),
        help='Path to PROGRESS.md (default: ~/Documents/lilyzhng/Learn/RL_PostTrain/PROGRESS.md)',
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT),
        help='Path to training_data.json (default: Qwen3-Coder/training_data.json)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be written without touching the file',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-parse and overwrite entries that already exist in training_data.json',
    )
    args = parser.parse_args()

    progress_path = Path(args.progress)
    output_path = Path(args.output)

    if not progress_path.exists():
        raise FileNotFoundError(f'PROGRESS.md not found: {progress_path}')

    md_text = progress_path.read_text(encoding='utf-8')
    sections = split_sections(md_text)
    print(f'Found {len(sections)} bug sections in PROGRESS.md')

    # Load existing entries
    existing: list[dict] = []
    existing_ids: set[str] = set()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding='utf-8'))
        existing_ids = {e['bug_id'] for e in existing}
        print(f'Existing training_data.json: {len(existing)} entries')

    new_entries: list[dict] = []
    skipped: list[str] = []

    for bug_id, title, body in sections:
        if bug_id in existing_ids and not args.force:
            skipped.append(bug_id)
            continue
        entry = section_to_entry(bug_id, title, body)
        new_entries.append(entry)
        commit_hint = f" (commit {entry['commit']})" if entry['commit'] else ''
        print(f'  + {bug_id}: {title}{commit_hint}')

    if skipped:
        print(f'\nSkipped {len(skipped)} already-present entries: {", ".join(skipped)}')
        print('  Pass --force to re-parse and overwrite them.')

    if not new_entries:
        print('\nNothing to write.')
        return

    if args.dry_run:
        print('\n--- DRY RUN (not writing) ---')
        print(json.dumps(new_entries, indent=2, ensure_ascii=False))
        return

    if args.force:
        # Replace existing entries that are being re-parsed
        new_ids = {e['bug_id'] for e in new_entries}
        kept = [e for e in existing if e['bug_id'] not in new_ids]
        all_entries = kept + new_entries
    else:
        all_entries = existing + new_entries

    output_path.write_text(
        json.dumps(all_entries, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'\nWrote {len(new_entries)} new entries → {output_path}')
    print(f'Total entries: {len(all_entries)}')


if __name__ == '__main__':
    main()
