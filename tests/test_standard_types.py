"""Regression: compose-api's runner builds its core from a module's ``standard_types`` table (``module:attribute``,
default attribute ``standard_types``) and calls ``.items()`` on it. This package shipped no such table, so a runner-style
core could not load its composite. Note: a plain tree alias for ``numeric_result`` did not append on repeated apply in
bigraph-schema 1.8.1 (probed), so this is a runner-contract fix, not a claim about append behaviour."""
from __future__ import annotations

import importlib

from process_bigraph import allocate_core


def test_standard_types_is_a_dict_naming_the_result_type():
    table = importlib.import_module("viva_copasi.types").standard_types
    assert isinstance(table, dict)
    assert "numeric_result" in table


def test_utc_step_composite_runs_on_a_runner_style_core():
    """Before: no standard_types, so the runner-style core lacked numeric_result and a bare allocate_core() failed
    with TypeError 'str' object does not support item assignment."""
    from viva_copasi.composites import build_composite
    from viva_copasi.core import register_processes

    core = allocate_core()
    for name, schema in importlib.import_module("viva_copasi.types").standard_types.items():
        core.register_type(name, schema)
    composite = build_composite("utc-step", core=register_processes(core))
    composite.run(1.0)
