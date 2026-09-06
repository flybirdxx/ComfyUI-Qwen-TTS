import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pause_utils import extract_leading_pause


class ExtractLeadingPauseTests(unittest.TestCase):
    def test_extracts_leading_pause(self):
        text, pause = extract_leading_pause("[break=5]Hello")

        self.assertEqual(text, "Hello")
        self.assertEqual(pause, 5.0)

    def test_combines_consecutive_leading_pauses(self):
        text, pause = extract_leading_pause(" [break=1.5] [break=2] Hello")

        self.assertEqual(text, "Hello")
        self.assertEqual(pause, 3.5)

    def test_accepts_case_insensitive_tag(self):
        text, pause = extract_leading_pause("[BREAK=0.5] Hello")

        self.assertEqual(text, "Hello")
        self.assertEqual(pause, 0.5)

    def test_does_not_extract_inline_pause(self):
        original = "Hello [break=5] world"

        text, pause = extract_leading_pause(original)

        self.assertEqual(text, original)
        self.assertEqual(pause, 0.0)


if __name__ == "__main__":
    unittest.main()
