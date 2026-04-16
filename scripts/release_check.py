#!/usr/bin/env python3
"""Local release gate checks for public readiness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def main() -> int:
    run([sys.executable, "-m", "compileall", str(ROOT / "src"), str(ROOT / "app.py"), str(ROOT / "scripts" / "demo_gradio.py")])
    run([sys.executable, str(ROOT / "app.py"), "--help"])
    run([sys.executable, str(ROOT / "scripts" / "demo_gradio.py"), "--help"])
    run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
