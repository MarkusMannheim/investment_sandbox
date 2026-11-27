"""The documents and docstrings must keep to the writing conventions, and a sentence that fails is rewritten."""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

TICS = {
    "em dash": "—",
    "antithesis reframe": r"(?i)\bnot (just|merely|only)\b|\bis not [^.,;]{3,40}[.,;] it is\b",
    "stock words": r"(?i)\b(delve|leverag\w*|robust|seamless|crucially|notably|tapestry|load-bearing)\b",
    "worth noting": r"(?i)\bworth (noting|stating|saying|remembering|comparing)\b",
    "the tell": r"(?i)\bthe tell\b",
    "flattery": r"(?i)\b(great question|good catch|you're right to)\b",
    "sentence opening And/But/So": r"(?m)(?<=[.!?])\s+(And|But|So)\s+[a-z]|^\s*(And|But|So)\s+[a-z]",
    "rhetorical question": r"(?m)^[^#\n`]*\?\s*$",
    # Calendar years only; a quoted line may carry a source's own title, and an ISO date is not a financial year
    "financial-year string": r"(?m)^(?!>)[^\n]*\b20[0-9]{2}-[0-9]{2}\b(?![0-9-])",
}

DOCS = ["README.md", "DATA_SOURCES.md", "DOCUMENTATION.md"]


def _sources():
    for name in DOCS:
        path = ROOT / name
        if path.exists():
            yield name, path.read_text()
    for nb in (ROOT / "notebooks").glob("*.ipynb"):
        cells = json.loads(nb.read_text())["cells"]
        yield nb.name, "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py")):
        yield str(path.relative_to(ROOT)), path.read_text()
    settings = ROOT / "src" / "investment_sandbox" / "data" / "settings.toml"
    yield "settings.toml", settings.read_text()


@pytest.mark.parametrize("tic,pattern", sorted(TICS.items()))
def test_the_prose_keeps_to_the_writing_conventions(tic, pattern):
    found = []
    for name, text in _sources():
        for m in re.finditer(pattern, text):
            line = text.count("\n", 0, m.start()) + 1
            excerpt = " ".join(text[max(0, m.start() - 40):m.start() + 50].split())
            found.append(f"{name}:{line}  ...{excerpt}...")
    assert not found, f"{len(found)} instance(s) of {tic}:\n  " + "\n  ".join(found[:12])
