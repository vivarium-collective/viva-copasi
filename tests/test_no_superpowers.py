"""Regression: the default path (build_core, build_composite, a run) must not need viva_superpowers.

viva_superpowers is dashboard tooling, git-only, and pins bigraph-schema>=1.8.2, which conflicts with the 1.8.1
core that compose-api's environment uses. Before this fix, build_core() raised ModuleNotFoundError when it was
absent. Runs in a subprocess that blocks the import, so other tests in this process cannot mask the failure.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import sys
    sys.modules["viva_superpowers"] = None  # any import of viva_superpowers now raises ImportError
    from viva_copasi.core import build_core
    from viva_copasi.composites import build_composite, list_composite_specs

    core = build_core()
    names = list_composite_specs()
    assert names, "no composite specs shipped"
    composite = build_composite(names[0], core=core)
    composite.run(0)
    print("ok", names[0])
    """
)


def test_default_path_loads_without_viva_superpowers():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().startswith("ok ")
