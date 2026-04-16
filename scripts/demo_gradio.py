#!/usr/bin/env python3
"""Compatibility wrapper for the packaged AksaraLLM Web UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aksarallm.webui import main


if __name__ == "__main__":
    raise SystemExit(main())
