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

# Issue #18 model: non-unit compartment volume (5) + a hasOnlySubstanceUnits
# species (S2), so concentration and amount differ and are distinguishable.
UNITS_MODEL = str(Path(__file__).parent / 'fixtures' / 'units_amount_conc.xml')

# Issue #27 model: a valid ODE model with NO species — only a parameter driven
# by a rate rule (dy/dt = 3). basico's get_species returns None for it.
NO_SPECIES_MODEL = str(Path(__file__).parent / 'fixtures' / 'no_species.xml')

# Issue #28 model: a one-species exponential decay (dS/dt = -k*S, k=0.5).
DECAY_MODEL = str(Path(__file__).parent / 'fixtures' / 'decay.xml')


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


# ---------------------------------------------------------------------------
# Issue #19 — remaining steady-state (Enhanced Newton) options
# ---------------------------------------------------------------------------

def test_copasi_steady_state_defaults_leave_method_options(core):
    """With no #19 keys set, the step leaves the model's stored Enhanced-Newton
    method settings untouched."""
    from basico import load_model

    baseline = load_model(TEST_MODEL)
    before = get_task_settings(T.STEADY_STATE, model=baseline)['method']

    step = CopasiSteadyStateStep(
        config={'model_source': TEST_MODEL},
        core=core,
    )
    step.update({})
    method = get_task_settings(T.STEADY_STATE, model=step.dm)['method']
    for key in (
        'Derivation Factor', 'Iteration Limit', 'Use Newton', 'Use Integration',
        'Use Back Integration', 'Accept Negative Concentrations',
        'Maximum duration for forward integration',
        'Maximum duration for backward integration',
    ):
        assert method[key] == before[key]


def test_copasi_steady_state_applies_method_options(core):
    """Each #19 key is applied to the Enhanced-Newton method settings."""
    step = CopasiSteadyStateStep(
        config={
            'model_source': TEST_MODEL,
            'derivation_factor': 1e-4,
            'iteration_limit': 77,
            'use_newton': False,
            'use_integration': True,
            'use_back_integration': True,
            'accept_negative_concentrations': True,
            'forward_integration_duration': 5e8,
            'backward_integration_duration': 2e5,
        },
        core=core,
    )
    step.update({})
    method = get_task_settings(T.STEADY_STATE, model=step.dm)['method']
    assert method['Derivation Factor'] == pytest.approx(1e-4)
    assert method['Iteration Limit'] == 77
    assert method['Use Newton'] is False
    assert method['Use Integration'] is True
    assert method['Use Back Integration'] is True
    assert method['Accept Negative Concentrations'] is True
    assert method['Maximum duration for forward integration'] == pytest.approx(5e8)
    assert method['Maximum duration for backward integration'] == pytest.approx(2e5)


def test_copasi_steady_state_options_coexist_with_tolerance_and_criterion(core):
    """#19 options apply alongside #14's relative_tolerance/criterion."""
    step = CopasiSteadyStateStep(
        config={
            'model_source': TEST_MODEL,
            'relative_tolerance': 1e-3,
            'criterion': 'Distance',
            'iteration_limit': 42,
            'use_integration': True,
        },
        core=core,
    )
    step.update({})
    method = get_task_settings(T.STEADY_STATE, model=step.dm)['method']
    assert method['Resolution'] == pytest.approx(1e-3)
    assert method['Target Criterion'] == 'Distance'
    assert method['Iteration Limit'] == 42
    assert method['Use Integration'] is True


# ---------------------------------------------------------------------------
# Issue #17 — output selection (choose the reported columns/elements)
# ---------------------------------------------------------------------------
# Repressilator (BIOMD0000000012) COPASI display names, used as selection
# strings. These mirror what viva-biomodels builds for the COPASI engine.
_SEL_SPECIES = '[LacI protein]'
_SEL_FLUX = '(degradation of LacI transcripts).Flux'


def test_copasi_utc_step_default_output_unchanged(core):
    """Absent `selections` keeps the default species+parameter output (#17)."""
    step = CopasiUTCStep(
        config={'model_source': TEST_MODEL, 'time': 5.0, 'n_points': 6},
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    # Default path reports SBML-id columns (species + parameters) and starts at 0.
    assert 'PX' in out['columns']
    assert out['time'][0] == 0.0
    assert len(out['time']) == 6


def test_copasi_utc_step_selections_restricts_columns(core):
    """`selections` yields exactly those output columns, in order (#17)."""
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'time': 5.0,
            'n_points': 6,
            'selections': ['Time', _SEL_SPECIES, _SEL_FLUX],
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    # 'Time' is lifted into out['time']; the rest become the output columns.
    assert out['columns'] == [_SEL_SPECIES, _SEL_FLUX]
    assert len(out['time']) == 6
    assert len(out['values']) == 6
    assert all(len(row) == 2 for row in out['values'])
    # A reaction flux is a non-default element now reported because it was asked
    # for — the behavior the default output could not provide.
    assert _SEL_FLUX in out['columns']


def test_copasi_utc_step_selections_honor_start_time(core):
    """`selections` still honors other options, e.g. start_time (#17 + #13)."""
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'time': 5.0,
            'n_points': 6,
            'start_time': 2.0,
            'selections': ['Time', _SEL_SPECIES],
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert out['time'][0] == pytest.approx(2.0)
    assert out['columns'] == [_SEL_SPECIES]


def test_copasi_steady_state_default_has_no_selections_key(core):
    """Absent `selections` leaves the steady-state output unchanged (#17)."""
    step = CopasiSteadyStateStep(
        config={'model_source': TEST_MODEL},
        core=core,
    )
    results = step.update({})['results']
    assert 'selections' not in results
    assert 'species_concentrations' in results
    assert 'fluxes' in results


def test_copasi_steady_state_selections_reports_values(core):
    """`selections` reports each requested element's steady-state value (#17)."""
    step = CopasiSteadyStateStep(
        config={
            'model_source': TEST_MODEL,
            'selections': [_SEL_SPECIES, _SEL_FLUX],
        },
        core=core,
    )
    results = step.update({})['results']
    assert set(results['selections']) == {_SEL_SPECIES, _SEL_FLUX}
    # One-element list per element (matches species_concentrations/fluxes shape).
    assert len(results['selections'][_SEL_SPECIES]) == 1
    assert isinstance(results['selections'][_SEL_SPECIES][0], float)


# ---------------------------------------------------------------------------
# Issue #15 — model_source may be raw SBML text, not just a URL/path. Mirrors
# the issue's repro: CopasiUTCStep({'model_source': open(...).read(), ...}).
# ---------------------------------------------------------------------------

def test_copasi_utc_step_accepts_sbml_text_as_model_source(core):
    """Raw SBML text as model_source loads and runs (no FileNotFoundError, #15)."""
    sbml_text = Path(TEST_MODEL).read_text(encoding='utf-8')
    step = CopasiUTCStep(
        config={'model_source': sbml_text, 'time': 5.0, 'n_points': 6},
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert len(out['time']) == 6
    assert len(out['columns']) > 0


def test_copasi_utc_step_path_still_works_unchanged(core):
    """A path model_source still loads and runs exactly as before (#15)."""
    step = CopasiUTCStep(
        config={'model_source': TEST_MODEL, 'time': 5.0, 'n_points': 6},
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert len(out['time']) == 6


def test_copasi_steady_state_accepts_sbml_text(core):
    """SteadyStateStep shares the loader, so it also accepts raw SBML text (#15)."""
    sbml_text = Path(TEST_MODEL).read_text(encoding='utf-8')
    step = CopasiSteadyStateStep(config={'model_source': sbml_text}, core=core)
    results = step.update({})['results']
    assert 'species_concentrations' in results


# ---------------------------------------------------------------------------
# Issue #16 — non-uniform output: output_times gives output at explicit points.
# Same key name/shape as the viva-tellurium sibling (#11). When set it overrides
# the uniform start_time/time/n_points path via basico's `values=` argument.
# ---------------------------------------------------------------------------

def test_copasi_utc_step_output_times_exact_points(core):
    """`output_times` yields rows at exactly the requested times (#16)."""
    times = [0.0, 0.1, 0.5, 2.0]
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'output_times': times,
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert out['time'] == pytest.approx(times)
    assert len(out['values']) == len(times)
    # Default species+parameter output is still produced (SBML-id columns).
    assert len(out['columns']) > 0


def test_copasi_utc_step_output_times_overrides_uniform(core):
    """`output_times` overrides start_time/time/n_points when both are given."""
    times = [0.0, 0.25, 1.0, 3.0, 7.0]
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'time': 5.0,          # would give a uniform grid...
            'n_points': 6,        # ...of 6 points ending at 5.0
            'start_time': 2.0,
            'output_times': times,  # ...but output_times wins
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert out['time'] == pytest.approx(times)
    assert len(out['values']) == len(times)


def test_copasi_utc_step_output_times_composes_with_selections(core):
    """`output_times` composes with `selections` (#16 + #17): exact times AND
    exactly the requested columns."""
    times = [0.0, 0.1, 0.5, 2.0]
    step = CopasiUTCStep(
        config={
            'model_source': TEST_MODEL,
            'output_times': times,
            'selections': ['Time', _SEL_SPECIES, _SEL_FLUX],
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert out['time'] == pytest.approx(times)
    assert out['columns'] == [_SEL_SPECIES, _SEL_FLUX]
    assert all(len(row) == 2 for row in out['values'])


def test_copasi_utc_step_default_output_unchanged_without_output_times(core):
    """Absent `output_times`, the uniform path is unchanged (no regression)."""
    step = CopasiUTCStep(
        config={'model_source': TEST_MODEL, 'time': 5.0, 'n_points': 6},
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert out['time'][0] == 0.0
    assert out['time'][-1] == pytest.approx(5.0)
    assert len(out['time']) == 6


# ---------------------------------------------------------------------------
# Issue #18 — species_units: concentration vs amount for the default species
# output. COPASI reports a species' bare name ("S1") as particle count but
# "[S1]" as concentration, and which one the default output uses depends on the
# species' hasOnlySubstanceUnits flag. The wrapper hides this: species_units
# ('concentration' | 'amount', default 'concentration') forces the default
# species output into one consistent kind for every species. Explicit
# `selections` are left verbatim and are unaffected. Same key name/semantics as
# the viva-tellurium sibling (#14).
# ---------------------------------------------------------------------------

def _units_ground_truth():
    """(concentration, amount) of the hasOnlySubstanceUnits species S2 at t=0,
    read straight from basico so the test asserts against COPASI's own numbers
    rather than hard-coded constants."""
    import basico
    dm = basico.load_model(UNITS_MODEL)
    dm.getModel().compileIfNecessary()
    sp = basico.get_species(model=dm)
    row = sp.loc[sp['sbml_id'] == 'S2'].iloc[0]
    return float(row['concentration']), float(row['particle_number'])


def test_copasi_utc_step_default_species_units_is_concentration(core):
    """Default (no species_units) reports concentration — the prior behavior."""
    conc, amount = _units_ground_truth()
    assert conc != pytest.approx(amount)  # volume != 1: the two are distinct

    step = CopasiUTCStep(
        config={'model_source': UNITS_MODEL, 'time': 5.0, 'n_points': 2},
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    s2 = out['columns'].index('S2')
    assert out['values'][0][s2] == pytest.approx(conc)


def test_copasi_utc_step_species_units_concentration_explicit(core):
    """species_units='concentration' reports the concentration of a
    hasOnlySubstanceUnits species (not its particle count)."""
    conc, amount = _units_ground_truth()
    step = CopasiUTCStep(
        config={
            'model_source': UNITS_MODEL, 'time': 5.0, 'n_points': 2,
            'species_units': 'concentration',
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    s2 = out['columns'].index('S2')
    assert out['values'][0][s2] == pytest.approx(conc)


def test_copasi_utc_step_species_units_amount(core):
    """species_units='amount' reports the particle count/amount of the same
    species — the conversion is observable because the compartment volume != 1."""
    conc, amount = _units_ground_truth()
    step = CopasiUTCStep(
        config={
            'model_source': UNITS_MODEL, 'time': 5.0, 'n_points': 2,
            'species_units': 'amount',
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    s2 = out['columns'].index('S2')
    assert out['values'][0][s2] == pytest.approx(amount)
    assert out['values'][0][s2] != pytest.approx(conc)


def test_copasi_steady_state_species_units_amount(core):
    """SteadyStateStep honors species_units for the default species output."""
    conc, amount = _units_ground_truth()

    ss_conc = CopasiSteadyStateStep(
        config={'model_source': UNITS_MODEL}, core=core,
    ).update({})['results']['species_concentrations']
    ss_amount = CopasiSteadyStateStep(
        config={'model_source': UNITS_MODEL, 'species_units': 'amount'},
        core=core,
    ).update({})['results']['species_concentrations']

    # At steady state the absolute values differ, but amount/concentration must
    # equal the fixed volume*Avogadro factor COPASI applies to S2 — proving the
    # 'amount' path returns particle number, not concentration.
    ratio = amount / conc
    assert ss_amount['S2'][0] / ss_conc['S2'][0] == pytest.approx(ratio, rel=1e-6)
    assert ss_amount['S2'][0] != pytest.approx(ss_conc['S2'][0])


def test_copasi_utc_step_selections_ignore_species_units(core):
    """Explicit `selections` are left VERBATIM: species_units does not rewrite
    or re-unit them (#17/#18 interaction)."""
    step = CopasiUTCStep(
        config={
            'model_source': UNITS_MODEL, 'time': 5.0, 'n_points': 2,
            'selections': ['Time', '[S2]'],
            'species_units': 'amount',
        },
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    # The requested column is reported exactly as asked, and '[S2]' is a
    # concentration selection — amount policy must not have touched it.
    conc, amount = _units_ground_truth()
    assert out['columns'] == ['[S2]']
    assert out['values'][0][0] == pytest.approx(conc)


# ---------------------------------------------------------------------------
# Issue #27 — a model with NO species must not crash. basico's get_species
# returns None for a valid ODE model that has only parameters + rate rules;
# interpret_sbml() must treat that as an empty species set and carry on.
# ---------------------------------------------------------------------------

def test_copasi_utc_step_no_species_model_loads(core):
    """A rate-rule-only model (no species) loads without TypeError (#27)."""
    step = CopasiUTCStep(
        config={'model_source': NO_SPECIES_MODEL, 'time': 1.0, 'n_points': 3},
        core=core,
    )
    # No species -> empty species set, not a crash.
    assert step.species_ids == []
    assert step.initial_state()['species_concentrations'] == {}


def test_copasi_utc_step_no_species_model_runs(core):
    """The no-species model still runs a time course and returns a result (#27)."""
    step = CopasiUTCStep(
        config={'model_source': NO_SPECIES_MODEL, 'time': 1.0, 'n_points': 3},
        core=core,
    )
    out = step.update({'species_counts': {}, 'species_concentrations': {}})['result']
    assert len(out['time']) == 3
    # The rate-rule variable y (dy/dt=3, y0=5) should reach y=8 at t=1.
    assert 'y' in out['columns']
    y = out['columns'].index('y')
    assert out['values'][-1][y] == pytest.approx(8.0, rel=1e-4)


def test_copasi_steady_state_step_no_species_model_loads(core):
    """SteadyStateStep shares interpret_sbml, so it also tolerates no species (#27)."""
    step = CopasiSteadyStateStep(
        config={'model_source': NO_SPECIES_MODEL},
        core=core,
    )
    assert step.species_ids == []


# ---------------------------------------------------------------------------
# Issue #28 — the UTC Step must be idempotent: a zero-time Step is a pure
# function of its inputs, so firing update() twice with the same inputs must
# return the same trajectory (it previously continued from the prior end-state).
# ---------------------------------------------------------------------------

def test_copasi_utc_step_is_idempotent(core):
    """Two identical updates return identical trajectories (#28)."""
    step = CopasiUTCStep(
        config={'model_source': DECAY_MODEL, 'time': 5.0, 'n_points': 2},
        core=core,
    )
    a = step.update({'species_counts': {}})['result']['values'][-1]
    b = step.update({'species_counts': {}})['result']['values'][-1]
    assert a == pytest.approx(b)


def test_copasi_utc_step_idempotent_with_selections(core):
    """Idempotency holds on the selections path too (#28)."""
    step = CopasiUTCStep(
        config={
            'model_source': DECAY_MODEL, 'time': 5.0, 'n_points': 2,
            'selections': ['Time', '[S]'],
        },
        core=core,
    )
    a = step.update({'species_counts': {}})['result']['values'][-1]
    b = step.update({'species_counts': {}})['result']['values'][-1]
    assert a == pytest.approx(b)


def test_copasi_steady_state_step_is_idempotent(core):
    """SteadyStateStep returns the same steady state on repeated firings (#28)."""
    step = CopasiSteadyStateStep(
        config={'model_source': TEST_MODEL},
        core=core,
    )
    a = step.update({})['results']['species_concentrations']
    b = step.update({})['results']['species_concentrations']
    for sid in a:
        assert a[sid][0] == pytest.approx(b[sid][0])


def test_copasi_utc_process_remains_stateful(core):
    """The Process variant is intentionally stateful across intervals: its
    reset-on-update must NOT be applied here (#28 guard)."""
    proc = CopasiUTCProcess(
        config={'model_source': DECAY_MODEL, 'time': 5.0, 'intervals': 1},
        core=core,
    )
    first = proc.update(proc.initial_state(), interval=5.0)['species_concentrations']['S']
    second = proc.update({}, interval=5.0)['species_concentrations']['S']
    # Decay continues from the previous end-state, so the second update is lower.
    assert second < first
