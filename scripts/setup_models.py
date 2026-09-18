#!/usr/bin/env python3
"""Standalone script to download and prepare Nido offline model assets."""

import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nido.config import load_config
from nido.models.manager import ModelManager


def main() -> int:
    config = load_config()
    manager = ModelManager(config)
    print("==================================================")
    print("   Nido Offline Model Asset Setup")
    print("==================================================")
    ok = manager.setup_all()
    if ok:
        print("\nAll models ready. You can now run 'nido'.")
        return 0
    else:
        print("\nSome models failed to install. Check log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
