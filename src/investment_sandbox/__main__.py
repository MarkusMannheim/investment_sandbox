"""One command to confirm a setup before the workshop: python -m investment_sandbox check."""

from __future__ import annotations

import sys
import time

from .config import load_settings
from .simulate import compare


def check() -> int:
    start = time.perf_counter()
    settings = load_settings()
    pair = compare(settings, seed=1, years=2)
    seconds = time.perf_counter() - start
    cautious, neutral = pair["cautious"], pair["risk_neutral"]
    print(f"Python {sys.version.split()[0]}, two years of three runs in {seconds:.1f} s")
    print(f"first year: {cautious.years[0].mean_price:.0f} $/MWh average price, "
          f"{len(cautious.years[0].builds)} blocks committed with caution, "
          f"{len(neutral.years[0].builds)} without")
    print("ready" if seconds < 30 else "slow: the notebook will still run, more slowly than planned")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["check"]:
        return check()
    print("usage: python -m investment_sandbox check")
    return 2


if __name__ == "__main__":
    sys.exit(main())
