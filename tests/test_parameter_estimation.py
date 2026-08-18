"""Tests for viva_copasi.parameter_estimation — COPASI parameter estimation.

Covers the functional wrapper (recover a broken parameter, and a
"which parameter is off" localization test) plus the ParameterEstimationStep
running through allocate_core() / register_link().

Test model: Elowitz 2000 Repressilator (BIOMD0000000012), vendored in
tests/fixtures/BIOMD0000000012_url.xml. Its fixed global quantities include
n = 2.0 (Hill coefficient) and KM = 40.0, which we break and recover.
"""
import warnings
from pathlib import Path

import pytest
import basico
from process_bigraph import allocate_core

from viva_copasi.parameter_estimation import (
    estimate,
    build_experiment_dataframe,
    normalize_fit_parameters,
    resimulation_rmsd,
)
from viva_copasi.processes import (
    CopasiUTCStep,
    CopasiSteadyStateStep,
    CopasiUTCProcess,
    ParameterEstimationStep,
)

warnings.filterwarnings("ignore")

TEST_MODEL = str(Path(__file__).parent / 'fixtures' / 'BIOMD0000000012_url.xml')
SPECIES = ['LacI protein', 'TetR protein', 'cI protein']
DUR, N = 100.0, 100


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('CopasiUTCStep', CopasiUTCStep)
    c.register_link('CopasiSteadyStateStep', CopasiSteadyStateStep)
    c.register_link('CopasiUTCProcess', CopasiUTCProcess)
    c.register_link('ParameterEstimationStep', ParameterEstimationStep)
    return c


def _reference_experiment(dm):
    """Simulate the reference (ground-truth) trace and return the experiment df."""
    ref = basico.run_time_course(duration=DUR, intervals=N, model=dm)
    return build_experiment_dataframe(ref, species=SPECIES)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def test_normalize_fit_parameters_wraps_bare_names():
    out = normalize_fit_parameters([
        {'name': 'n', 'lower': 0.5, 'upper': 4.0, 'start': 1.0},
        {'name': 'Values[KM]', 'lower': 1.0, 'upper': 100.0, 'start': 40.0},
    ])
    assert out[0]['name'] == 'Values[n]'          # bare name wrapped
    assert out[1]['name'] == 'Values[KM]'         # already-qualified left alone
    # bounds untouched
    assert out[0]['lower'] == 0.5 and out[0]['start'] == 1.0


# ---------------------------------------------------------------------------
# (a) recover a broken parameter
# ---------------------------------------------------------------------------

def test_estimate_recovers_broken_hill_coefficient():
    """Break n (2.0 -> 1.0), fit it, and recover it; re-simulated RMSD ~ 0."""
    dm = basico.load_model(TEST_MODEL)
    true_n = float(basico.get_parameters(model=dm).loc['n', 'value'])
    assert true_n == pytest.approx(2.0)

    experiment = _reference_experiment(dm)

    # Break n; the simulation honors the set even though get_parameters read-back lags.
    basico.set_parameters(name='n', exact=True, initial_value=1.0, model=dm)

    result = estimate(
        dm, experiment,
        [{'name': 'n', 'lower': 0.5, 'upper': 4.0, 'start': 1.0}],
    )

    assert 'Values[n]' in result['fitted']
    assert result['fitted']['Values[n]'] == pytest.approx(2.0, abs=1e-3)
    assert result['objective'] < 1e-6
    assert result['rms'] < 1e-3

    # Verify by re-simulation rather than trusting get_parameters read-back.
    rmsd = resimulation_rmsd(result['model'], experiment, species=SPECIES)
    assert rmsd < 1e-3


def test_estimate_recovers_broken_km():
    """Break KM (40.0 -> 10.0), fit it back, verify by resimulation."""
    dm = basico.load_model(TEST_MODEL)
    experiment = _reference_experiment(dm)

    basico.set_parameters(name='KM', exact=True, initial_value=10.0, model=dm)

    result = estimate(
        dm, experiment,
        [{'name': 'KM', 'lower': 1.0, 'upper': 200.0, 'start': 10.0}],
    )

    assert result['fitted']['Values[KM]'] == pytest.approx(40.0, rel=1e-2)
    rmsd = resimulation_rmsd(result['model'], experiment, species=SPECIES)
    assert rmsd < 1e-2


# ---------------------------------------------------------------------------
# (b) "which parameter is off" — localize the truly-broken parameter
# ---------------------------------------------------------------------------

def test_which_parameter_is_off_broken_one_moves_most():
    """Fit several candidates at once; the truly-broken one moves most from start.

    Only n is broken (2.0 -> 1.0). We fit n, KM and tps_active together, each
    starting from its current model value. The broken parameter (n) should move
    the most (largest relative displacement from its start), localizing the fault.
    """
    dm = basico.load_model(TEST_MODEL)
    experiment = _reference_experiment(dm)

    # Break ONLY n.
    basico.set_parameters(name='n', exact=True, initial_value=1.0, model=dm)

    starts = {'n': 1.0, 'KM': 40.0, 'tps_active': 0.5}
    fit_params = [
        {'name': 'n', 'lower': 0.5, 'upper': 4.0, 'start': starts['n']},
        {'name': 'KM', 'lower': 1.0, 'upper': 200.0, 'start': starts['KM']},
        {'name': 'tps_active', 'lower': 0.01, 'upper': 5.0, 'start': starts['tps_active']},
    ]

    # Levenberg-Marquardt is the standard local least-squares method; the
    # default global method is unreliable on this oscillatory 3-parameter fit.
    result = estimate(dm, experiment, fit_params, method='Levenberg - Marquardt')

    # Relative movement of each fitted parameter from its start value.
    movement = {}
    for bare, start in starts.items():
        fitted = result['fitted'][f'Values[{bare}]']
        movement[bare] = abs(fitted - start) / max(abs(start), 1e-9)

    assert movement['n'] == max(movement.values()), movement
    # And the untouched parameters barely move.
    assert movement['KM'] < 0.1
    assert movement['tps_active'] < 0.1
    # The fit reproduces the reference.
    assert resimulation_rmsd(result['model'], experiment, species=SPECIES) < 1e-2


# ---------------------------------------------------------------------------
# (c) ParameterEstimationStep through the core
# ---------------------------------------------------------------------------

def test_parameter_estimation_step_recovers_via_core(core):
    """The Step runs through allocate_core()/register_link and produces fitted outputs.

    Uses param_overrides to break n after the reference is generated, then fits it.
    """
    step = ParameterEstimationStep(
        config={
            'model_source': TEST_MODEL,
            'reference_species': SPECIES,
            'reference_duration': DUR,
            'reference_intervals': N,
            'param_overrides': {'n': 1.0},
            'fit_params': [
                {'name': 'n', 'lower': 0.5, 'upper': 4.0, 'start': 1.0},
            ],
        },
        core=core,
    )

    out = step.update({})

    assert 'fitted' in out and 'objective' in out and 'rms' in out
    assert 'resim_rmsd' in out and 'solution' in out

    assert out['fitted']['Values[n]'] == pytest.approx(2.0, abs=1e-3)
    assert out['objective'] < 1e-6
    assert out['resim_rmsd'] < 1e-3
    assert isinstance(out['solution'], list) and len(out['solution']) == 1


def test_parameter_estimation_step_requires_fit_params(core):
    """Constructing the Step without fit_params raises."""
    with pytest.raises(ValueError):
        ParameterEstimationStep(
            config={'model_source': TEST_MODEL, 'fit_params': []},
            core=core,
        )
