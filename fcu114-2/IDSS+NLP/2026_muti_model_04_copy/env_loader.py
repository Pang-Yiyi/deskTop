"""Minimal .env loader for local demo secrets.

This avoids adding a python-dotenv dependency. Values already present in the
process environment are not overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(env_path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ if the file exists."""
    path = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
