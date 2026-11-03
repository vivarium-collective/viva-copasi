"""Regression: compose-api's runner builds its core from a module's ``standard_types`` table (``module:attribute``,
default attribute ``standard_types``) and calls ``.items()`` on it. This package shipped no such table, so a runner-style
core could not load its composite. Note: a plain tree alias for ``numeric_result`` did not append on repeated apply in
bigraph-schema 1.8.1 (probed), so this is a runner-contract fix, not a claim about append behaviour."""
from __future__ import annotations

import importlib

import pytest
from process_bigraph import allocate_core

from viva_copasi.composites import build_composite, list_composite_specs
from viva_copasi.core import register_processes


def test_standard_types_is_a_dict_naming_the_result_type():
    table = importlib.import_module("viva_copasi.types").standard_types
    assert isinstance(table, dict)
    assert "numeric_result" in table


@pytest.mark.parametrize("name", list_composite_specs())
def test_every_shipped_composite_runs_on_a_runner_style_core(name):
    """Every shipped composite must load and run on the core compose-api's runner builds: allocate_core() plus this
    package's standard_types, with no build_core(). Regression: steady-state failed with TypeError 'str' object does
    not support item assignment, because CopasiSteadyStateStep's 'any' output was missing from standard_types."""
    core = allocate_core()
    for type_name, schema in importlib.import_module("viva_copasi.types").standard_types.items():
        core.register_type(type_name, schema)
    composite = build_composite(name, core=register_processes(core))
    composite.run(1.0)
