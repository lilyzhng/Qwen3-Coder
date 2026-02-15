import pathlib
import unittest

from finetuning.ui_metrics import (
    HTMLValidator,
    compute_color_entropy,
    extract_color_tokens,
    extract_tailwind_classes,
)


class TestHTMLValidator(unittest.TestCase):
    def test_valid_html(self):
        validator = HTMLValidator()
        validator.feed("<div><span>OK</span></div>")
        self.assertTrue(validator.is_valid())
        self.assertEqual(validator.errors, [])

    def test_mismatched_tags(self):
        validator = HTMLValidator()
        validator.feed("<div><span>Oops</div>")
        self.assertFalse(validator.is_valid())
        self.assertTrue(any("Tag mismatch" in err for err in validator.errors))

    def test_unexpected_closing_tag(self):
        validator = HTMLValidator()
        validator.feed("</p>")
        self.assertFalse(validator.is_valid())
        self.assertTrue(
            any("Unexpected closing tag" in err for err in validator.errors)
        )

    def test_pseudo_generated_html_metrics(self):
        fixture_path = (
            pathlib.Path(__file__).parent / "fixtures" / "pseudo_generated.html"
        )
        html = fixture_path.read_text(encoding="utf-8")

        validator = HTMLValidator()
        validator.feed(html)
        self.assertTrue(validator.is_valid())

        colors = extract_color_tokens(html)
        classes = extract_tailwind_classes(html)
        entropy = compute_color_entropy(colors)

        # Basic sanity checks for UI metrics
        self.assertGreater(len(classes), 0)
        self.assertGreater(len(colors), 0)
        self.assertGreaterEqual(entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
