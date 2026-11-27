"""The model's answers on one seed must not move unless tools/write_golden.py is run again on purpose."""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from write_golden import GOLDEN, fingerprint  # noqa: E402


@pytest.mark.skipif(not GOLDEN.exists(), reason="no fingerprint written yet")
def test_the_numbers_have_not_moved():
    want = json.loads(GOLDEN.read_text())
    got = fingerprint()
    for leg in ("risk_neutral", "cautious", "tender"):
        for key, value in want[leg].items():
            assert got[leg][key] == pytest.approx(value, rel=1e-6, abs=1e-3), (
                f"{leg}.{key} moved. If that was meant, regenerate with tools/write_golden.py")
