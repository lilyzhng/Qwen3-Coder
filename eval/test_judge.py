"""Local test harness for the eval judge.

Tests JSON parsing logic (no API key needed) and judge scoring (needs OPENROUTER_API_KEY).

Usage:
    # Unit tests only (no API key):
    python -m pytest Qwen3-Coder/eval/test_judge.py -v -k "parse"

    # Integration tests (needs OPENROUTER_API_KEY):
    python -m pytest Qwen3-Coder/eval/test_judge.py -v -k "judge"

    # All tests:
    python -m pytest Qwen3-Coder/eval/test_judge.py -v
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import extract_json_from_response from the eval script.
#
# modal_eval_instruct.py defines this function inside run_evaluation(),
# so we can't import it directly. Instead, we replicate the exact same
# function here and test it. When it's stable, we can refactor the eval
# script to import from a shared module.
# ---------------------------------------------------------------------------


def extract_json_from_response(raw_content: str) -> dict | None:
    """Extract a JSON object from a judge response using brace-counting.

    Handles: clean JSON, ```json fences, text preamble before JSON,
    escaped quotes inside strings, and nested structures.
    Returns the parsed dict or None if no valid JSON found.
    """
    if not raw_content or not raw_content.strip():
        return None

    text = raw_content.strip()

    # Strip markdown fences if present (anywhere in the text)
    text = re.sub(r'```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*', '', text)

    # Find the first '{' and extract JSON by brace counting
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    # Sanitize newlines inside string values (common in judge reasoning)
    def sanitize_string_values(s):
        result = []
        in_str = False
        esc = False
        for c in s:
            if esc:
                result.append(c)
                esc = False
                continue
            if c == '\\' and in_str:
                result.append(c)
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                result.append(c)
                continue
            if in_str and c in ('\n', '\r'):
                result.append(' ')
                continue
            result.append(c)
        return ''.join(result)

    def try_parse(json_str):
        json_str = sanitize_string_values(json_str)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and 'score' in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    if depth == 0:
        # Clean extraction — the JSON was complete
        json_str = text[start:end]
        return try_parse(json_str)

    # Truncated JSON — the judge response was cut off.
    # Try to repair by closing open strings/arrays/objects.
    json_str = text[start:]

    # Attempt 1: close the string + close braces/brackets
    repair = json_str.rstrip().rstrip(',')
    if in_string:
        repair += '"'
    # Close any open arrays, then objects
    # Count open brackets that aren't closed
    bracket_depth = 0
    str_mode = False
    esc_mode = False
    for c in repair:
        if esc_mode:
            esc_mode = False
            continue
        if c == '\\' and str_mode:
            esc_mode = True
            continue
        if c == '"':
            str_mode = not str_mode
            continue
        if str_mode:
            continue
        if c == '[':
            bracket_depth += 1
        elif c == ']':
            bracket_depth -= 1
    repair += ']' * max(bracket_depth, 0)
    repair += '}' * depth

    result = try_parse(repair)
    if result:
        return result

    # Attempt 2: truncate to the last complete key-value pair
    # Find last comma outside a string and close there
    last_comma = -1
    str_mode = False
    esc_mode = False
    for idx, c in enumerate(json_str):
        if esc_mode:
            esc_mode = False
            continue
        if c == '\\' and str_mode:
            esc_mode = True
            continue
        if c == '"':
            str_mode = not str_mode
            continue
        if str_mode:
            continue
        if c == ',':
            last_comma = idx

    if last_comma > 0:
        truncated = json_str[:last_comma]
        # Close any open arrays
        bracket_depth = 0
        str_mode = False
        esc_mode = False
        for c in truncated:
            if esc_mode:
                esc_mode = False
                continue
            if c == '\\' and str_mode:
                esc_mode = True
                continue
            if c == '"':
                str_mode = not str_mode
                continue
            if str_mode:
                continue
            if c == '[':
                bracket_depth += 1
            elif c == ']':
                bracket_depth -= 1
        truncated += ']' * max(bracket_depth, 0)
        truncated += '}'
        result = try_parse(truncated)
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Fixtures path
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent.parent / 'wandb' / 'eval_results' / \
    'comparison-Qwen3-Coder-Next-FP8-vs-Qwen3-Coder-Next-sft-r8-attn-20260220-074051-0226-0411'

SAMPLE_IDS = ['376', '589', '649', '680', '737']


def fixture_path(filename: str) -> Path:
    return FIXTURES_DIR / filename


# ---------------------------------------------------------------------------
# extract_code — replicated from modal_eval_instruct.py
# ---------------------------------------------------------------------------

def extract_code(response_text: str) -> str:
    """Extract HTML/code from a model response, stripping markdown fences and preamble."""
    # Try matching complete fenced code blocks first
    pattern = r"```(?:html|css|tsx|jsx|vue)?\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    # Handle truncated output: opening fence but no closing fence
    # (common when model hits max_new_tokens before closing ```)
    open_pattern = r"```(?:html|css|tsx|jsx|vue)?\s*\n(.*)"
    open_match = re.search(open_pattern, response_text, re.DOTALL)
    if open_match:
        return open_match.group(1).strip()
    stripped = response_text.strip()
    if stripped.startswith("<") or stripped.startswith("<!"):
        return stripped
    return stripped


# ===========================================================================
# Unit Tests — extract_code (no API calls)
# ===========================================================================

class TestExtractCode:
    """Test extract_code handles complete, truncated, and bare code."""

    def test_complete_fence(self):
        raw = '```html\n<!DOCTYPE html>\n<html><body>Hello</body></html>\n```'
        result = extract_code(raw)
        assert result.startswith('<!DOCTYPE html>')
        assert '```' not in result

    def test_fence_with_preamble(self):
        raw = 'Here is a complete, single-file solution.\n\n```html\n<!DOCTYPE html>\n<html><body>Timer</body></html>\n```'
        result = extract_code(raw)
        assert result.startswith('<!DOCTYPE html>')
        assert 'single-file solution' not in result

    def test_truncated_fence_no_closing(self):
        """Model hit max_new_tokens before closing ```. Should still extract code."""
        raw = 'Here is the solution:\n\n```html\n<!DOCTYPE html>\n<html>\n<head><title>Timer</title></head>\n<body>\n<div>Pomodoro</div>'
        result = extract_code(raw)
        assert result.startswith('<!DOCTYPE html>')
        assert 'Here is the solution' not in result
        assert '```' not in result

    def test_truncated_fence_with_long_preamble(self):
        """Real-world case: model outputs explanation + code, hits token limit."""
        raw = """Here is a complete, single-file solution containing HTML, CSS, and JavaScript.

This app features a "Pomodoro-style" workflow (Focus / Short Break / Long Break) with a smooth, animated circular progress bar and a pastel color palette.

### How to use this:
1. Copy the code block below.
2. Create a new file named `timer.html`.
3. Open it in any web browser.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pastel Focus Timer</title>
    <style>
        body { font-family: sans-serif; }
    </style>
</head>
<body>
<div class="timer">25:00</div>"""
        result = extract_code(raw)
        assert result.startswith('<!DOCTYPE html>')
        assert 'How to use this' not in result
        assert 'Pomodoro-style' not in result

    def test_bare_html_no_fence(self):
        """Output is pure HTML without any markdown fences."""
        raw = '<!DOCTYPE html>\n<html><body>Hello</body></html>'
        result = extract_code(raw)
        assert result.startswith('<!DOCTYPE html>')

    def test_gt_with_fences(self):
        """Ground truth answer wrapped in ```html...``` fences."""
        raw = '```html\n<!DOCTYPE html>\n<html class="bg-pastel">\n<body>Timer</body>\n</html>\n```'
        result = extract_code(raw)
        assert result.startswith('<!DOCTYPE html>')
        assert '```' not in result


# ===========================================================================
# Unit Tests — JSON Parsing (no API calls)
# ===========================================================================

class TestParseCleanJson:
    """Test parsing clean JSON with no wrapping."""

    def test_basic(self):
        raw = '{"score": 8, "failure_modes": [], "reasoning": "Good UI"}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 8
        assert result['failure_modes'] == []

    def test_with_whitespace(self):
        raw = '  \n  {"score": 7, "failure_modes": ["generic-colors"], "reasoning": "OK"}  \n  '
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 7


class TestParseFencedJson:
    """Test parsing JSON wrapped in ```json fences."""

    def test_basic_fence(self):
        raw = '```json\n{"score": 8, "failure_modes": [], "reasoning": "Good"}\n```'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 8

    def test_fence_no_language(self):
        raw = '```\n{"score": 6, "failure_modes": ["missing-states"], "reasoning": "No hover"}\n```'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 6

    def test_fence_with_trailing_newlines(self):
        raw = '```json\n{"score": 9, "failure_modes": [], "reasoning": "Great"}\n```\n\n'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 9


class TestParseTextBeforeFence:
    """Test parsing when judge puts text before the JSON fence."""

    def test_text_then_fence(self):
        raw = 'Here is my evaluation:\n\n```json\n{"score": 5, "failure_modes": ["broken-layout"], "reasoning": "Overlap"}\n```'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 5

    def test_multiline_preamble(self):
        raw = """Based on the provided code and screenshot, here is the evaluation:

**1. Framework:** The code uses Tailwind CSS correctly.
**2. Layout:** Some issues with overlap.

```json
{"score": 4, "failure_modes": ["broken-code", "wrong-framework", "broken-layout"], "reasoning": "10 - 4 (broken) - 3 (layout) = 3, rounded to 4"}
```"""
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 4
        assert 'broken-code' in result['failure_modes']

    def test_text_then_bare_json(self):
        """Judge returns text analysis then bare JSON (no fences)."""
        raw = 'The code shows a timer app with good design.\n\n{"score": 7, "failure_modes": ["missing-states"], "reasoning": "No hover effects"}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 7


class TestParseEscapedQuotes:
    """Test parsing responses with escaped quotes in reasoning."""

    def test_escaped_quotes_in_reasoning(self):
        raw = '{"score": 6, "failure_modes": ["generic-colors"], "reasoning": "Uses \\"default\\" Tailwind colors without customization"}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 6
        assert 'default' in result['reasoning']

    def test_newlines_in_reasoning(self):
        raw = '{"score": 5, "failure_modes": ["broken-layout"], "reasoning": "Issues found:\n- Elements overlap\n- Wrong alignment"}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 5


class TestParseNoJson:
    """Test graceful fallback when response has no JSON."""

    def test_pure_text(self):
        raw = 'The generated solution fails on two major technical requirements: it does not use the requested framework.'
        result = extract_json_from_response(raw)
        assert result is None

    def test_empty_response(self):
        result = extract_json_from_response('')
        assert result is None

    def test_none_response(self):
        result = extract_json_from_response(None)
        assert result is None

    def test_json_without_score(self):
        raw = '{"failure_modes": ["broken-code"], "reasoning": "Bad"}'
        result = extract_json_from_response(raw)
        assert result is None  # Missing 'score' field


class TestParseTruncatedJson:
    """Test parsing truncated JSON (judge response cut off mid-sentence)."""

    def test_truncated_reasoning(self):
        """Reasoning string cut off — no closing quote or brace."""
        raw = '{"score": 4, "failure_modes": ["wrong-framework", "broken-code"], "reasoning": "The model failed to use Tailwind CSS as required, opting for vanilla CSS instead. The generated code includes conversational text'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 4
        assert 'wrong-framework' in result['failure_modes']

    def test_truncated_after_comma(self):
        """Response cut off right after a comma."""
        raw = '{"score": 6, "failure_modes": ["wrong-framework", "broken-layout", "broken-code"], "reasoning": "The generation fails to use Tailwind CSS (visible \'</\' in the nav),'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 6

    def test_truncated_in_array(self):
        """Response cut off inside the failure_modes array."""
        raw = '{"score": 3, "failure_modes": ["broken-code", "wrong-framework'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 3

    def test_truncated_minimal(self):
        """Only score is complete — should still parse."""
        raw = '{"score": 7, "failure_modes": ['
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 7


class TestParseRealFailures:
    """Test against actual failure cases from eval runs."""

    def test_680_fenced_json_that_failed(self):
        """This was an actual judge response that caused judge-error.

        The regex r"\\{[^{}]*(?:\\[[^\\[\\]]*\\][^{}]*)*\\}" couldn't parse it
        because the reasoning contained complex text.
        """
        raw = '```json\n{"score": 4, "failure_modes": ["broken-code", "wrong-framework", "broken-layout"], "reasoning": "The generated code is severely truncated, ending in the middle of CSS definitions. The browser renders a blank page (-4). No Tailwind CSS used (-2)."}\n```'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 4
        assert 'broken-code' in result['failure_modes']

    def test_589_text_before_json(self):
        """Judge returned text analysis before any JSON."""
        raw = """Based on the provided code and screenshot, here is the evaluation:

The rubric expected Tailwind CSS but the generation uses vanilla CSS with custom properties.

```json
{"score": 3, "failure_modes": ["broken-code", "wrong-framework"], "reasoning": "10 - 4 (broken) - 2 (wrong framework) - 1 (missing states) = 3"}
```"""
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 3

    def test_nested_braces_in_reasoning(self):
        """Reasoning that mentions code with braces."""
        raw = '{"score": 7, "failure_modes": [], "reasoning": "The code uses CSS variables like :root { --bg: #fff } which is good practice"}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result['score'] == 7


# ===========================================================================
# Integration Tests — Judge API (needs OPENROUTER_API_KEY)
# ===========================================================================

def _get_openrouter_client():
    """Create OpenRouter client or skip if no API key."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        pytest.skip('OPENROUTER_API_KEY not set')

    from openai import OpenAI
    return OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key)


def _load_fixture_raw(sample_id: str, variant: str = 'base') -> str:
    """Load raw model output from fixture files."""
    path = fixture_path(f'{sample_id}_{variant}_raw.txt')
    if not path.exists():
        pytest.skip(f'Fixture not found: {path}')
    return path.read_text()


def _load_fixture_gt(sample_id: str) -> str:
    """Load ground truth HTML from fixture files."""
    path = fixture_path(f'{sample_id}_gt.html')
    if not path.exists():
        pytest.skip(f'Fixture not found: {path}')
    return path.read_text()


def _load_fixture_screenshot(sample_id: str, variant: str) -> str | None:
    """Load screenshot path if it exists."""
    path = fixture_path(f'screenshots/{sample_id}_{variant}.png')
    return str(path) if path.exists() else None


# Import the judge prompt from the eval script (read it to stay in sync)
def _get_judge_prompt() -> str:
    """Read the JUDGE_PROMPT from modal_eval_instruct.py to stay in sync."""
    eval_script = Path(__file__).parent / 'modal_eval_instruct.py'
    content = eval_script.read_text()

    # Extract the JUDGE_PROMPT string
    match = re.search(r'JUDGE_PROMPT = """\\\n(.*?)"""', content, re.DOTALL)
    if not match:
        pytest.fail('Could not extract JUDGE_PROMPT from modal_eval_instruct.py')
    return match.group(1)


def _call_judge(client, prompt: str, model_output: str, reference: str,
                gen_img: str | None, gt_img: str | None,
                judge_model: str = 'google/gemini-3-pro-preview') -> dict:
    """Call the judge API and parse the response.

    This replicates the judge_output logic from modal_eval_instruct.py
    but uses the fixed extract_json_from_response.
    """
    import base64
    import time

    judge_prompt = _get_judge_prompt()
    judge_text = judge_prompt.format(
        prompt=prompt,
        model_output=model_output[:8000],
        reference=reference[:4000],
    )
    content_parts = [{'type': 'text', 'text': judge_text}]

    def img_to_b64(path):
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return None

    gen_b64 = img_to_b64(gen_img)
    if gen_b64:
        content_parts.append({'type': 'text', 'text': '\n\nGeneration screenshot:'})
        content_parts.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{gen_b64}'},
        })

    gt_b64 = img_to_b64(gt_img)
    if gt_b64:
        content_parts.append({'type': 'text', 'text': '\n\nGround truth screenshot:'})
        content_parts.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{gt_b64}'},
        })

    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(1)

            response = client.chat.completions.create(
                model=judge_model,
                messages=[{'role': 'user', 'content': content_parts}],
                max_tokens=2048,
                temperature=0.0,
            )
            raw_content = response.choices[0].message.content
            print(f'  [attempt {attempt+1}] Raw judge response ({len(raw_content or "")} chars):')

            if not raw_content or not raw_content.strip():
                last_error = 'Empty response'
                continue

            parsed = extract_json_from_response(raw_content)
            if parsed:
                return parsed

            last_error = f'No JSON in response: {raw_content[:100]}...'
            continue

        except Exception as e:
            last_error = f'API error: {e}'
            continue

    return {'score': 0, 'failure_modes': ['judge-error'], 'reasoning': f'After {max_retries+1} attempts: {last_error}'}


@pytest.mark.integration
class TestJudgeTimer737:
    """Timer sample 737 — good UI with working circular countdown.

    The model generates a complete Pomodoro timer with SVG progress ring,
    mode switching, and keyboard shortcuts. Screenshot shows a working UI.
    Previous judge scored 4/10 due to truncated code — with the fixed prompt
    and 8000 char limit, should score 6+.
    """

    def test_judge_timer_base(self):
        client = _get_openrouter_client()
        raw = _load_fixture_raw('737', 'base')
        gt = _load_fixture_gt('737')
        gen_img = _load_fixture_screenshot('737', 'base')
        gt_img = _load_fixture_screenshot('737', 'gt')

        # Use a generic prompt (we don't have the original question in fixtures)
        prompt = 'Create a Pomodoro timer with a minimalist, calming design'

        result = _call_judge(client, prompt, raw, gt, gen_img, gt_img)
        print(f'\n  Timer 737 (base) score: {result["score"]}/10')
        print(f'  Failure modes: {result.get("failure_modes", [])}')
        print(f'  Reasoning: {result.get("reasoning", "")}')

        assert result['score'] > 0, f'Judge error: {result.get("reasoning")}'
        assert 'judge-error' not in result.get('failure_modes', []), \
            f'JSON parsing failed: {result.get("reasoning")}'
        # Timer renders a working UI — should score at least 6
        assert result['score'] >= 6, \
            f'Timer scored {result["score"]}/10, expected 6+. Reasoning: {result.get("reasoning")}'

    def test_judge_timer_lora(self):
        client = _get_openrouter_client()
        raw = _load_fixture_raw('737', 'lora')
        gt = _load_fixture_gt('737')
        gen_img = _load_fixture_screenshot('737', 'lora')
        gt_img = _load_fixture_screenshot('737', 'gt')

        prompt = 'Create a Pomodoro timer with a minimalist, calming design'

        result = _call_judge(client, prompt, raw, gt, gen_img, gt_img)
        print(f'\n  Timer 737 (lora) score: {result["score"]}/10')
        print(f'  Failure modes: {result.get("failure_modes", [])}')
        print(f'  Reasoning: {result.get("reasoning", "")}')

        assert result['score'] > 0, f'Judge error: {result.get("reasoning")}'
        assert 'judge-error' not in result.get('failure_modes', [])


@pytest.mark.integration
class TestJudgeIntranet376:
    """Intranet sample 376 — broken output (text description, not code).

    The model outputs a MERN stack design doc instead of HTML/CSS.
    Should score low (1-3).
    """

    def test_judge_intranet_base(self):
        client = _get_openrouter_client()
        raw = _load_fixture_raw('376', 'base')
        gt = _load_fixture_gt('376')
        gen_img = _load_fixture_screenshot('376', 'base')
        gt_img = _load_fixture_screenshot('376', 'gt')

        prompt = 'Create a corporate intranet portal with employee directory and messaging'

        result = _call_judge(client, prompt, raw, gt, gen_img, gt_img)
        print(f'\n  Intranet 376 (base) score: {result["score"]}/10')
        print(f'  Failure modes: {result.get("failure_modes", [])}')
        print(f'  Reasoning: {result.get("reasoning", "")}')

        assert result['score'] > 0, f'Judge error: {result.get("reasoning")}'
        assert 'judge-error' not in result.get('failure_modes', [])
        # Broken output — should score low
        assert result['score'] <= 4, \
            f'Intranet scored {result["score"]}/10, expected <=4 (broken output)'


@pytest.mark.integration
class TestJudgeAllSamplesParseOk:
    """Verify the judge returns parseable JSON for all 5 samples.

    This catches the 50%+ judge-error rate from the original regex parser.
    With the brace-counting parser, all responses should parse correctly.
    """

    @pytest.mark.parametrize('sample_id', SAMPLE_IDS)
    def test_no_judge_error(self, sample_id):
        client = _get_openrouter_client()

        raw_path = fixture_path(f'{sample_id}_base_raw.txt')
        if not raw_path.exists():
            pytest.skip(f'No fixture for {sample_id}')
        raw = raw_path.read_text()

        gt_path = fixture_path(f'{sample_id}_gt.html')
        gt = gt_path.read_text() if gt_path.exists() else ''

        gen_img = _load_fixture_screenshot(sample_id, 'base')
        gt_img = _load_fixture_screenshot(sample_id, 'gt')

        prompt = f'UI generation task (sample {sample_id})'

        result = _call_judge(client, prompt, raw, gt, gen_img, gt_img)
        print(f'\n  Sample {sample_id} score: {result["score"]}/10')

        assert 'judge-error' not in result.get('failure_modes', []), \
            f'Sample {sample_id}: JSON parsing failed — {result.get("reasoning")}'
        assert 1 <= result['score'] <= 10, \
            f'Sample {sample_id}: Score {result["score"]} out of range'
