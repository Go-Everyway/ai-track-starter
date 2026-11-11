"""lib/figures.py writes and merges named figures."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from lib import figures


class FiguresTests(unittest.TestCase):
    def test_record_merges_and_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["FIGURES_DIR"] = d
            try:
                figures.record("m99", a=0.123456789, n=63)
                out = figures.record("m99", b=[1.0, 2.5])
                self.assertEqual(out, {"a": 0.123457, "b": [1.0, 2.5], "n": 63})
                self.assertEqual(json.loads((Path(d) / "m99.json").read_text()), out)
            finally:
                del os.environ["FIGURES_DIR"]


if __name__ == "__main__":
    unittest.main()
