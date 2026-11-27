"""The notebook is built by a script and has to run.

The committed notebook must be what tools/build_notebook.py produces, so the two
cannot drift apart. Running it end to end is slow and happens under the slow marker,
which continuous integration runs on one Python version.
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_notebook import NOTEBOOK, notebook  # noqa: E402


@pytest.fixture(scope="module")
def committed():
    return json.loads(NOTEBOOK.read_text())


def test_the_committed_notebook_is_the_built_one(committed):
    assert committed == notebook(), "run python tools/build_notebook.py and commit the result"


def test_the_notebook_says_what_it_is_and_is_not(committed):
    text = "\n".join("".join(c["source"]) for c in committed["cells"] if c["cell_type"] == "markdown")
    assert "illustrative" in text.lower()
    assert "one region" in text.lower()


def test_the_notebook_sets_no_tasks(committed):
    text = "\n".join("".join(c["source"]) for c in committed["cells"] if c["cell_type"] == "markdown")
    for word in ("exercise", "task", "your turn", "try to"):
        assert word not in text.lower(), f"the notebook explains and invites; it sets no {word}"


@pytest.mark.slow
def test_the_notebook_runs_end_to_end(committed, tmp_path):
    import nbformat
    from nbclient import NotebookClient

    # Reading through nbformat joins each cell's lines, which the kernel needs
    nb = nbformat.reads(json.dumps(committed), as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    errors = [out for cell in nb.cells if cell.cell_type == "code"
              for out in cell.get("outputs", []) if out.get("output_type") == "error"]
    assert not errors, errors[0]
