"""The README's Python sample must stay real: extract it and execute it verbatim.

Charter rule: no README claim without a committed test. This makes the code
sample and its gate the same bytes — a renamed API or a fictional scene path
fails here before it can mislead a reader.
"""

import os
import re

import pytest

lpw = pytest.importorskip("latentphysics")
pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_snippet_runs_verbatim(monkeypatch):
    monkeypatch.chdir(ROOT)  # the snippet uses a repo-relative scene path
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        src = f.read()
    blocks = re.findall(r"```python\n(.*?)```", src, re.S)
    assert len(blocks) == 1, "README should carry exactly one python sample"
    ns = {}
    exec(compile(blocks[0], "README.md#snippet", "exec"), ns)
    obs = ns["obs"]
    assert obs.is_cuda and obs.shape[0] == 4096
    assert torch.isfinite(obs).all()
