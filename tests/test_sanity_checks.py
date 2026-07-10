"""Smoke test for the sanity-check guard-rails script.

Runs `scripts/sanity_checks.py` as a subprocess (not imported), so it exercises
exactly the same code path a developer or CI would run by hand.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sanity_checks.py"


def test_sanity_checks_script_passes():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert result.returncode == 0, (
        f"sanity_checks.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "ALL CHECKS PASSED" in result.stdout
