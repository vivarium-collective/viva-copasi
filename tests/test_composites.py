"""Integration tests for viva-copasi composite specs.

Tests cover:
- Discovery: list_composite_specs() returns the three expected names.
- Structural validity: each *.composite.yaml has the required top-level keys.
- Instantiation: build_composite() produces a runnable process_bigraph.Composite.
- Short run: the UTC process composite advances and emits data.

Test model: Elowitz 2000 Repressilator (BIOMD0000000012), bundled at
viva_copasi/composites/repressilator.xml.
"""
from pathlib import Path

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from viva_copasi.composites import (
    list_composite_specs,
    load_composite_spec,
    build_composite,
    register_copasi,
    make_copasi_utc_process_document,
    make_copasi_steady_state_document,
    _COMPOSITES_DIR,
)
from viva_copasi.processes import (
    CopasiUTCStep,
    CopasiSteadyStateStep,
    CopasiUTCProcess,
)

# Repressilator model — same file bundled in the composites package dir.
BUNDLED_MODEL = str(_COMPOSITES_DIR / "repressilator.xml")


@pytest.fixture
def core():
    """Minimal core with all COPASI classes + RAM emitter registered."""
    return register_copasi()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_list_composite_specs_returns_three():
    """Discovery returns the three expected composite names."""
    specs = list_composite_specs()
    assert 'utc-step' in specs
    assert 'utc-process' in specs
    assert 'steady-state' in specs


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------

def test_utc_step_spec_well_formed():
    """utc-step.composite.yaml has required top-level keys."""
    spec = load_composite_spec('utc-step')
    assert 'name' in spec
    assert 'state' in spec
    assert spec['name'] == 'utc-step'
    assert 'copasi_utc' in spec['state']
    assert spec['state']['copasi_utc']['address'] == 'local:CopasiUTCStep'


def test_utc_process_spec_well_formed():
    """utc-process.composite.yaml has required top-level keys."""
    spec = load_composite_spec('utc-process')
    assert 'name' in spec
    assert 'state' in spec
    assert spec['name'] == 'utc-process'
    assert 'copasi_utc_process' in spec['state']
    assert spec['state']['copasi_utc_process']['address'] == 'local:CopasiUTCProcess'


def test_steady_state_spec_well_formed():
    """steady-state.composite.yaml has required top-level keys."""
    spec = load_composite_spec('steady-state')
    assert 'name' in spec
    assert 'state' in spec
    assert spec['name'] == 'steady-state'
    assert 'copasi_ss' in spec['state']
    assert spec['state']['copasi_ss']['address'] == 'local:CopasiSteadyStateStep'


# ---------------------------------------------------------------------------
# build_composite — instantiation
# ---------------------------------------------------------------------------

def test_build_composite_utc_step(core):
    """build_composite('utc-step') instantiates a Composite without error."""
    sim = build_composite('utc-step', core=core)
    assert sim is not None
    assert isinstance(sim, Composite)


def test_build_composite_utc_process(core):
    """build_composite('utc-process') instantiates a Composite without error."""
    sim = build_composite('utc-process', core=core)
    assert sim is not None
    assert isinstance(sim, Composite)


def test_build_composite_steady_state(core):
    """build_composite('steady-state') instantiates a Composite without error."""
    sim = build_composite('steady-state', core=core)
    assert sim is not None
    assert isinstance(sim, Composite)


# ---------------------------------------------------------------------------
# End-to-end: UTC process composite — short run + emitter check
# ---------------------------------------------------------------------------

def test_utc_process_composite_short_run(core):
    """CopasiUTCProcess composite runs 2 steps and emits species_concentrations."""
    doc = make_copasi_utc_process_document(
        model_source=BUNDLED_MODEL,
        time=1.0,
        intervals=5,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)

    stores = sim.state['stores']
    assert 'species_concentrations' in stores
    assert isinstance(stores['species_concentrations'], dict)
    # Repressilator has 6 species
    assert len(stores['species_concentrations']) >= 3


def test_utc_process_composite_emitter(core):
    """Emitter collects at least one record after a short run."""
    doc = make_copasi_utc_process_document(
        model_source=BUNDLED_MODEL,
        time=1.0,
        intervals=5,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)

    raw = gather_emitter_results(sim)
    rows = raw[('emitter',)]
    assert len(rows) >= 1
    # Each row should have species_concentrations
    row = rows[0]
    assert 'species_concentrations' in row or 'fluxes' in row


# ---------------------------------------------------------------------------
# End-to-end: steady-state composite — single step
# ---------------------------------------------------------------------------

def test_steady_state_composite_runs(core):
    """CopasiSteadyStateStep composite runs and populates results store."""
    doc = make_copasi_steady_state_document(model_source=BUNDLED_MODEL)
    sim = Composite({'state': doc}, core=core)
    sim.run(1.0)

    ss_stores = sim.state['ss_stores']
    assert 'results' in ss_stores


# ---------------------------------------------------------------------------
# Issue #25 — a document using the COPASI steps must load under the documented
# build_core() core. The step output types ('numeric_result', 'any') are not in
# bigraph-schema's BASE_TYPES; build_core() must register them (as register_copasi
# does) so composite wiring resolves instead of failing with a confusing
# "TypeError: 'str' object does not support item assignment".
# ---------------------------------------------------------------------------

def test_build_core_registers_output_types():
    """build_core() registers the step output schema types (#25)."""
    from viva_copasi.core import build_core

    core = build_core()
    assert 'numeric_result' in core.registry
    assert 'any' in core.registry


def test_build_core_loads_utc_step_document():
    """A CopasiUTCStep document loads under build_core() without error (#25)."""
    from process_bigraph import Composite
    from viva_copasi.core import build_core

    model = str(_COMPOSITES_DIR / "repressilator.xml")
    doc = {
        "state": {
            "w": {
                "_type": "step",
                "address": "local:!viva_copasi.processes.CopasiUTCStep",
                "config": {"model_source": model, "time": 5.0, "n_points": 6},
                "inputs": {
                    "species_concentrations": ["a"],
                    "species_counts": ["b"],
                },
                "outputs": {"result": ["r"]},
            }
        }
    }
    sim = Composite(doc, core=build_core())
    assert isinstance(sim, Composite)


def test_build_core_loads_steady_state_document():
    """A CopasiSteadyStateStep document (output type 'any') loads too (#25)."""
    from process_bigraph import Composite
    from viva_copasi.core import build_core

    model = str(_COMPOSITES_DIR / "repressilator.xml")
    doc = {
        "state": {
            "ss": {
                "_type": "step",
                "address": "local:!viva_copasi.processes.CopasiSteadyStateStep",
                "config": {"model_source": model, "time": 0.0},
                "inputs": {
                    "species_concentrations": ["a"],
                    "counts": ["b"],
                },
                "outputs": {"results": ["r"]},
            }
        }
    }
    sim = Composite(doc, core=build_core())
    assert isinstance(sim, Composite)
