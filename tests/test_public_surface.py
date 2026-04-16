from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class PublicSurfaceTests(unittest.TestCase):
    def test_package_import_without_runtime_dependencies(self):
        sys.path.insert(0, str(SRC))
        import aksarallm

        self.assertEqual(aksarallm.__version__, "2.0.0")
        cfg = aksarallm.get_config("1.5b")
        self.assertEqual(cfg.max_seq_len, 32768)

    def test_cli_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "app.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Web UI untuk AksaraLLM.", result.stdout)

    def test_demo_wrapper_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "demo_gradio.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model MODEL", result.stdout)

    def test_no_hardcoded_hf_tokens_in_repo_sources(self):
        matches = []
        flagged_prefix = "hf_" + "PLACEHOLDER_TOKEN_FOR_TESTING"
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if "data" in path.parts and "generated" in path.parts:
                continue
            if path.name == "test_public_surface.py":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if flagged_prefix in text:
                matches.append(str(path.relative_to(ROOT)))
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
