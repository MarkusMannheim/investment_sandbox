"""Pin the model's answers on one seed, so that a change which moves them is one someone meant to make."""

import json
import pathlib

from investment_sandbox.config import load_settings
from investment_sandbox.simulate import compare

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "tests" / "golden" / "fingerprint.json"
SEED, YEARS = 1, 20


def fingerprint() -> dict:
    pair = compare(load_settings(), seed=SEED, years=YEARS)
    out = {"seed": SEED, "years": YEARS}
    for name, r in pair.items():
        out[name] = {
            "unserved_mwh": [round(y.unserved_mwh, 3) for y in r.years],
            "mean_price": [round(y.mean_price, 4) for y in r.years],
            "built_mw": [round(sum(y.built_mw.values()), 1) for y in r.years],
            "bought_mw": [round(sum(y.bought_mw.values()), 1) for y in r.years],
            "bill": round(r.bill, 0),
            "resource_cost": round(r.resource_cost, 0),
            "underwriting_cost": round(r.underwriting_cost, 0),
            "expected_underwriting_cost": round(r.expected_underwriting_cost, 0),
        }
    return out


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(fingerprint(), indent=1) + "\n")
    print(f"wrote {GOLDEN}")
