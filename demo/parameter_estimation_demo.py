"""Demo: recover a broken model parameter with COPASI parameter estimation.

Runs offline against the vendored Repressilator model (BIOMD0000000012). It:

  1. simulates a reference time course from the true model,
  2. breaks the Hill coefficient n (2.0 -> 1.0),
  3. fits n back to the reference with viva_copasi.parameter_estimation.estimate,
  4. verifies the recovery by re-simulation (RMSD vs the reference),
  5. repeats as a "which parameter is off?" localization over 3 candidates.

Run:
    cd <viva-copasi worktree>
    PYTHONPATH=$PWD .venv/bin/python demo/parameter_estimation_demo.py
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import basico

from viva_copasi.parameter_estimation import (
    estimate,
    build_experiment_dataframe,
    resimulation_rmsd,
)

MODEL = str(
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "BIOMD0000000012_url.xml"
)
SPECIES = ["LacI protein", "TetR protein", "cI protein"]
DUR, N = 100.0, 100


def recover_one_parameter():
    print("=" * 68)
    print("Demo 1 — recover a single broken parameter (Hill coefficient n)")
    print("=" * 68)

    dm = basico.load_model(MODEL)
    true_n = float(basico.get_parameters(model=dm).loc["n", "value"])
    print(f"  true n = {true_n}")

    # Reference trace from the true model.
    ref = basico.run_time_course(duration=DUR, intervals=N, model=dm)
    experiment = build_experiment_dataframe(ref, species=SPECIES)

    # Break n (the simulation honors the set even if get_parameters read-back lags).
    basico.set_parameters(name="n", exact=True, initial_value=1.0, model=dm)
    print("  broke n -> 1.0, now fitting it back against the reference ...")

    result = estimate(
        dm, experiment,
        [{"name": "n", "lower": 0.5, "upper": 4.0, "start": 1.0}],
    )

    recovered = result["fitted"]["Values[n]"]
    rmsd = resimulation_rmsd(result["model"], experiment, species=SPECIES)
    print(f"  recovered n     = {recovered:.6f}")
    print(f"  fit objective   = {result['objective']:.3e}")
    print(f"  fit rms         = {result['rms']:.3e}")
    print(f"  resimulation RMSD vs reference = {rmsd:.3e}  (near 0 => good fit)")


def which_parameter_is_off():
    print()
    print("=" * 68)
    print("Demo 2 — which parameter is off? (fit 3 candidates, only n is broken)")
    print("=" * 68)

    dm = basico.load_model(MODEL)
    ref = basico.run_time_course(duration=DUR, intervals=N, model=dm)
    experiment = build_experiment_dataframe(ref, species=SPECIES)

    # Break ONLY n.
    basico.set_parameters(name="n", exact=True, initial_value=1.0, model=dm)

    starts = {"n": 1.0, "KM": 40.0, "tps_active": 0.5}
    fit_params = [
        {"name": "n", "lower": 0.5, "upper": 4.0, "start": starts["n"]},
        {"name": "KM", "lower": 1.0, "upper": 200.0, "start": starts["KM"]},
        {"name": "tps_active", "lower": 0.01, "upper": 5.0, "start": starts["tps_active"]},
    ]

    # Levenberg-Marquardt is the reliable local least-squares method here.
    result = estimate(dm, experiment, fit_params, method="Levenberg - Marquardt")

    print("  parameter    start      fitted     rel. movement")
    print("  " + "-" * 48)
    ranked = []
    for bare, start in starts.items():
        fitted = result["fitted"][f"Values[{bare}]"]
        move = abs(fitted - start) / max(abs(start), 1e-9)
        ranked.append((move, bare, start, fitted))
        print(f"  {bare:<11} {start:<10.4g} {fitted:<10.4g} {move:.4f}")

    ranked.sort(reverse=True)
    culprit = ranked[0][1]
    print(f"\n  => the parameter that moved most is '{culprit}' (the broken one).")


if __name__ == "__main__":
    recover_one_parameter()
    which_parameter_is_off()
