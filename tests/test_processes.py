"""Tests for viva_copasi.processes — UTC Step, SteadyState Step, UTC Process.

Ports the logic of biocompose's run_copasi_utc() and run_copasi_ss()
into proper pytest tests, plus adds coverage for CopasiUTCProcess.

Test model: Elowitz 2000 Repressilator (BIOMD0000000012), vendored in
tests/fixtures/BIOMD0000000012_url.xml.
"""
import pytest
from pathlib import Path

from process_bigraph import allocate_core
from basico import get_task_settings, T

from viva_copasi.processes import (
    BaseCopasi,
    CopasiUTCStep,
    CopasiSteadyStateStep,
    CopasiUTCProcess,
)

# Repressilator model — resolved relative to this file so tests run from any cwd.
TEST_MODEL = str(Path(__file__).parent / 'fixtures' / 'BIOMD0000000012_url.xml')


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('CopasiUTCStep', CopasiUTCStep)
    c.register_link('CopasiSteadyStateStep', CopasiSteadyStateStep)
    c.register_link('CopasiUTCProcess', CopasiUTCProcess)
    return c


# ---------------------------------------------------------------------------
# CopasiUTCStep
# ---------------------------------------------------------------------------

def test_copasi_utc_step_runs(core):
    """Port of biocompose's run_copasi_utc(): produces a non-empty trajectory."""
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'time': 10.0,
            'n_points': 5,
        },
        core=core,
    )

    initial = step.initial_state()
    assert 'species_concentrations' in initial
    assert isinstance(initial['species_concentrations'], dict)
    assert len(initial['species_concentrations']) > 0

    result = step.update(initial)

    assert result is not None
    assert 'result' in result

    tc = result['result']
    assert 'time' in tc
    assert 'columns' in tc
    assert 'values' in tc

    # 5 n_points → 4 intervals → 5 rows (including t=0)
    assert len(tc['time']) == 5
    assert len(tc['values']) == 5
    # Columns should include species ids
    assert len(tc['columns']) > 0


def test_copasi_utc_step_initial_state_has_all_species(core):
    """initial_state() returns concentrations for all model species."""
    step = CopasiUTCStep(
        config={'model_source': TEST_MODEL, 'time': 1.0, 'n_points': 2},
        core=core,
    )
    initial = step.initial_state()
    species = initial['species_concentrations']
    # Repressilator has 6 species (lacI, tetR, cI, pLacI, pTetR, pCI)
    assert len(species) >= 3
    for k, v in species.items():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# CopasiSteadyStateStep
# ---------------------------------------------------------------------------

def test_copasi_steady_state_step_returns_concentrations(core):
    """Port of biocompose's run_copasi_ss(): returns species concentrations at steady state."""
    step = CopasiSteadyStateStep(
        config={'model_source': TEST_MODEL},
        core=core,
    )

    initial = step.initial_state()
    assert 'species_concentrations' in initial

    result = step.update(initial)

    assert result is not None
    assert 'results' in result

    results = result['results']
    assert 'time' in results
    assert 'species_concentrations' in results
    assert 'fluxes' in results

    # One-point time series (t=0)
    assert results['time'] == [0.0]

    # Each species has exactly one concentration value
    for sid, vals in results['species_concentrations'].items():
        assert isinstance(vals, list)
        assert len(vals) == 1
        assert isinstance(vals[0], float)

    # At least one reaction flux recorded
    assert len(results['fluxes']) > 0


def test_copasi_steady_state_step_empty_inputs(core):
    """SteadyStateStep runs successfully with an empty inputs dict."""
    step = CopasiSteadyStateStep(
        config={'model_source': TEST_MODEL},
        core=core,
    )
    result = step.update({})
    assert 'results' in result


# ---------------------------------------------------------------------------
# CopasiUTCProcess
# ---------------------------------------------------------------------------

def test_copasi_utc_process_initial_state(core):
    """initial_state() returns species concentrations keyed by SBML ID."""
    proc = CopasiUTCProcess(
        config={'model_source': TEST_MODEL, 'time': 1.0, 'intervals': 5},
        core=core,
    )
    initial = proc.initial_state()
    assert 'species_concentrations' in initial
    assert len(initial['species_concentrations']) > 0


def test_copasi_utc_process_advances_one_step(core):
    """The Process variant produces species_concentrations, fluxes, and time after one update."""
    proc = CopasiUTCProcess(
        config={'model_source': TEST_MODEL, 'time': 1.0, 'intervals': 5},
        core=core,
    )
    initial = proc.initial_state()

    out = proc.update(initial, interval=1.0)

    assert out is not None
    assert 'species_concentrations' in out
    assert 'fluxes' in out
    assert 'time' in out

    # species_concentrations: dict of SBML-ID -> float (final state, not list)
    for sid, val in out['species_concentrations'].items():
        assert isinstance(val, float)

    # fluxes: dict of reaction-id -> float
    assert len(out['fluxes']) > 0

    # time: list of floats from the run_time_course call
    assert isinstance(out['time'], list)
    assert len(out['time']) > 0


def test_copasi_utc_process_multiple_steps(core):
    """Consecutive updates advance the simulation state."""
    proc = CopasiUTCProcess(
        config={'model_source': TEST_MODEL, 'time': 1.0, 'intervals': 5},
        core=core,
    )
    state = proc.initial_state()

    out1 = proc.update(state, interval=1.0)
    out2 = proc.update(out1, interval=1.0)

    # Both updates return valid output
    assert out2 is not None
    assert 'species_concentrations' in out2


# ---------------------------------------------------------------------------
# Issue #13 — start_time is honored (was hard-coded to 0.0)
# ---------------------------------------------------------------------------

def test_copasi_utc_step_defaults_to_start_time_zero(core):
    """Absent start_time preserves prior behavior: the trace begins at t=0."""
    step = CopasiUTCStep(
        config={'model_source': TEST_MODEL, 'time': 5.0, 'n_points': 6},
        core=core,
    )
    tc = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert tc['time'][0] == 0.0


def test_copasi_utc_step_honors_start_time(core):
    """start_time != 0 actually shifts where the time course begins (#13)."""
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'time': 5.0,
            'n_points': 6,
            'start_time': 2.0,
        },
        core=core,
    )
    tc = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    # First output time point is the configured start_time, not 0.0.
    assert tc['time'][0] == pytest.approx(2.0)


def test_copasi_utc_process_honors_start_time(core):
    """The Process variant also honors start_time (#13)."""
    proc = CopasiUTCProcess(
        config={
            'model_source': TEST_MODEL,
            'time': 1.0,
            'intervals': 5,
            'start_time': 3.0,
        },
        core=core,
    )
    out = proc.update(proc.initial_state(), interval=5.0)
    assert out['time'][0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Issue #14 — method / tolerance / step-size are wired through to basico
# ---------------------------------------------------------------------------

def test_copasi_utc_step_defaults_leave_basico_defaults(core):
    """Absent option keys leave basico's own defaults in place (no regression)."""
    step = CopasiUTCStep(
        config={'model_source': TEST_MODEL, 'time': 5.0, 'n_points': 6},
        core=core,
    )
    step.update({'species_counts': {}, 'species_concentrations': {}})
    method = get_task_settings(T.TIME_COURSE, model=step.dm)['method']
    # basico's default integrator is deterministic LSODA with r_tol 1e-6.
    assert 'LSODA' in method['name']
    assert method['Relative Tolerance'] == pytest.approx(1e-6)


def test_copasi_utc_step_applies_method(core):
    """A non-default method is applied to the COPASI task (#14)."""
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'time': 5.0,
            'n_points': 6,
            'method': 'stochastic',
        },
        core=core,
    )
    step.update({'species_counts': {}, 'species_concentrations': {}})
    name = get_task_settings(T.TIME_COURSE, model=step.dm)['method']['name']
    assert 'Stochastic' in name


def test_copasi_utc_step_applies_tolerances(core):
    """Non-default relative/absolute tolerances are applied to the task (#14)."""
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'time': 5.0,
            'n_points': 6,
            'relative_tolerance': 1e-3,
            'absolute_tolerance': 1e-9,
        },
        core=core,
    )
    step.update({'species_counts': {}, 'species_concentrations': {}})
    method = get_task_settings(T.TIME_COURSE, model=step.dm)['method']
    assert method['Relative Tolerance'] == pytest.approx(1e-3)
    assert method['Absolute Tolerance'] == pytest.approx(1e-9)


def test_copasi_utc_process_applies_relative_tolerance(core):
    """The Process variant also wires relative_tolerance through (#14)."""
    proc = CopasiUTCProcess(
        config={
            'model_source': TEST_MODEL,
            'time': 1.0,
            'intervals': 5,
            'relative_tolerance': 1e-3,
        },
        core=core,
    )
    proc.update(proc.initial_state(), interval=1.0)
    method = get_task_settings(T.TIME_COURSE, model=proc.dm)['method']
    assert method['Relative Tolerance'] == pytest.approx(1e-3)


def test_copasi_steady_state_applies_resolution_and_criterion(core):
    """SteadyStateStep wires relative_tolerance (Resolution) and criterion (#14)."""
    step = CopasiSteadyStateStep(
        config={
            'model_source': TEST_MODEL,
            'relative_tolerance': 1e-3,
            'criterion': 'Distance',
        },
        core=core,
    )
    step.update({})
    method = get_task_settings(T.STEADY_STATE, model=step.dm)['method']
    assert method['Resolution'] == pytest.approx(1e-3)
    assert method['Target Criterion'] == 'Distance'
