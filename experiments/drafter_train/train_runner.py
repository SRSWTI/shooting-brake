#!/usr/bin/env python3
"""Register the Laguna draft implementation, then enter SpecForge's typed CLI."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import laguna_dflash_model  # noqa: E402,F401
from specforge.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
