#!/usr/bin/env python3
"""Self-contained test runner for Nido using standard Python."""

import sys
import traceback
from pathlib import Path

# Add src to pythonpath
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tests.test_config
import tests.test_tools
import tests.test_laya
import tests.test_action_candidates
import tests.test_accessibility
import tests.test_desktop_agent
import tests.test_stt
import tests.test_audio
import tests.test_hotkey
import tests.test_pipeline
import tests.test_cli


def run_all() -> int:
    modules = [
        ("Config Tests", tests.test_config),
        ("Tools Tests", tests.test_tools),
        ("Laya Decision Agent Tests", tests.test_laya),
        ("Action Candidate Builder Tests", tests.test_action_candidates),
        ("Accessibility Perception Tests", tests.test_accessibility),
        ("Desktop Multi-Step Agent Tests", tests.test_desktop_agent),
        ("STT Normalization Tests", tests.test_stt),
        ("Audio Processing Tests", tests.test_audio),
        ("Hotkey Push-to-Talk Tests", tests.test_hotkey),
        ("Pipeline Orchestration Tests", tests.test_pipeline),
        ("CLI Commands Tests", tests.test_cli),
    ]

    total = 0
    passed = 0
    failed = 0

    print("=" * 60)
    print("        Nido Self-Check Test Runner")
    print("=" * 60)

    for suite_name, mod in modules:
        print(f"\n--- {suite_name} ---")
        for attr in dir(mod):
            if attr.startswith("test_"):
                fn = getattr(mod, attr)
                if callable(fn):
                    total += 1
                    try:
                        fn()
                        print(f"  [PASS] {attr}")
                        passed += 1
                    except Exception as e:
                        print(f"  [FAIL] {attr}: {e}")
                        traceback.print_exc()
                        failed += 1

    print("\n" + "=" * 60)
    print(f"Summary: {passed}/{total} passed, {failed} failed.")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
