#!/usr/bin/env python3
"""Local launcher for the packaged AksaraLLM Gradio app."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aksarallm.webui import main


if __name__ == "__main__":
    raise SystemExit(main())
