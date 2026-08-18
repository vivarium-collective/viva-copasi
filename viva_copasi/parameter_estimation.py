"""Parameter estimation via COPASI's Parameter Estimation task.

This module is a thin, AI-free functional wrapper around `basico`'s parameter
estimation API. It fits model parameters so that a simulated time course matches
a reference (experimental) trace. Under the hood it drives COPASI's
*Parameter Estimation* task: it registers the reference trace as a fitting
experiment (`basico.add_experiment`), declares the parameters to estimate as fit
items (`basico.set_fit_parameters`), and runs the optimizer
(`basico.run_parameter_estimation`). See the COPASI docs for the task itself:
http://copasi.org/Support/User_Manual/Tasks/Parameter_Estimation/

Design notes / API quirks (already discovered, encoded here so callers don't
have to rediscover them):

- A fitting experiment DataFrame needs an explicit ``"Time"`` *column*, but
  ``basico.run_time_course`` returns Time as the *index*. Use
  :func:`build_experiment_dataframe` to convert.
- Global quantities are fit-addressed as ``Values[<name>]``. A bare ``<name>``
  silently no-ops ("object <name> not found"). :func:`estimate` normalizes bare
  names for you.
- ``basico.get_parameters()`` value read-back is unreliable across
  ``set_parameters`` (a known basico quirk), but the *simulation* honors the set
  value. Verify a recovered fit by re-simulating and comparing RMSD
  (:func:`resimulation_rmsd`), not by trusting ``get_parameters``.
"""
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import basico

__all__ = [
    'build_experiment_dataframe',
    'normalize_fit_parameters',
    'estimate',
    'resimulation_rmsd',
]


def _resolve_model_source(model_source: str) -> str:
    """Resolve a model reference to a loadable path or URL.

    URLs pass through unchanged; relative paths resolve against ``Path.cwd()``.
    Mirrors ``viva_copasi.processes._model_path_resolution``.
    """
    if model_source.startswith(('http://', 'https://')):
        return model_source
    p = Path(model_source)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def _as_datamodel(model: Union[str, Any]):
    """Return a loaded COPASI DataModel.

    ``model`` may be a path/URL string (loaded via ``basico.load_model``) or an
    already-loaded DataModel object, which is returned unchanged.
    """
    if isinstance(model, str):
        dm = basico.load_model(_resolve_model_source(model))
        if dm is None:
            raise RuntimeError(
                f"load_model({model!r}) returned None. "
                "Check that the file exists and is a valid COPASI/SBML model."
            )
        return dm
    return model


def build_experiment_dataframe(
    time_course: pd.DataFrame,
    species: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a fitting-experiment DataFrame from a ``run_time_course`` result.

    ``basico.run_time_course`` returns Time as the DataFrame *index*, but a
    COPASI fitting experiment needs an explicit ``"Time"`` *column*. This helper
    performs that conversion.

    Args:
        time_course: DataFrame as returned by ``basico.run_time_course`` (Time in
            the index, one column per observed species).
        species: Optional subset of columns to keep as observables. Defaults to
            every column of ``time_course``.

    Returns:
        A DataFrame with a leading ``"Time"`` column followed by the species
        columns, suitable for ``basico.add_experiment`` / :func:`estimate`.
    """
    df = pd.DataFrame({"Time": np.asarray(time_course.index.to_numpy(), dtype=float)})
    cols = list(species) if species is not None else list(time_course.columns)
    for c in cols:
        df[c] = np.asarray(time_course[c].to_numpy(), dtype=float)
    return df


def _normalize_fit_name(name: str) -> str:
    """Normalize a fit-item name to COPASI's ``Values[<name>]`` addressing.

    Accepts an already-qualified reference (``Values[n]``, ``Values[n].InitialValue``,
    ``{Reaction}.k1``, ...) unchanged, and wraps a bare global-quantity name
    (``n``) as ``Values[n]``.
    """
    name = name.strip()
    # Already an explicit object reference (contains a bracket or a scoped '.').
    if '[' in name or name.startswith(('Values', 'Compartments', '(')):
        return name
    return f"Values[{name}]"


def normalize_fit_parameters(
    fit_params: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a copy of ``fit_params`` with each item's ``name`` normalized.

    Each item is a dict with keys ``name`` (bare or ``Values[...]``), ``lower``,
    ``upper`` and ``start``. Only ``name`` is rewritten; other keys pass through.
    """
    out = []
    for item in fit_params:
        item = dict(item)
        item['name'] = _normalize_fit_name(item['name'])
        out.append(item)
    return out


def _load_experiment(experiment: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """Coerce an experiment argument into a DataFrame.

    Accepts a DataFrame directly, or a path to a delimited-text file (``.csv``,
    ``.tsv``/``.txt`` treated as tab/whitespace-separated).
    """
    if isinstance(experiment, pd.DataFrame):
        return experiment
    path = _resolve_model_source(str(experiment))
    suffix = Path(path).suffix.lower()
    if suffix in ('.tsv', '.txt'):
        return pd.read_csv(path, sep=None, engine='python')
    return pd.read_csv(path)


def estimate(
    model_source: Union[str, Any],
    experiment: Union[pd.DataFrame, str],
    fit_params: List[Dict[str, Any]],
    *,
    method: Optional[str] = None,
    update_model: bool = True,
    experiment_name: str = "reference",
) -> Dict[str, Any]:
    """Estimate model parameters by fitting a reference time course.

    Wraps COPASI's Parameter Estimation task: registers ``experiment`` as a
    fitting experiment, declares ``fit_params`` as fit items, and runs the
    optimizer.

    Args:
        model_source: Path/URL to an SBML or COPASI model, or an already-loaded
            COPASI DataModel.
        experiment: The reference trace to fit against — a DataFrame with a
            ``"Time"`` column plus one column per observed species (see
            :func:`build_experiment_dataframe`), or a path to such a CSV/TSV.
        fit_params: List of fit items, each ``{"name", "lower", "upper", "start"}``.
            ``name`` may be a bare global-quantity name (``"n"``) — it is
            normalized to ``Values[n]``.
        method: Optional COPASI optimization method name (e.g.
            ``"Levenberg - Marquardt"``). ``None`` uses COPASI's current default.
        update_model: If True (default), write the fitted values back into the
            model so a subsequent simulation uses them.
        experiment_name: Name to register the fitting experiment under.

    Returns:
        A dict with:
          - ``fitted``: ``{normalized_name: fitted_value}``
          - ``objective``: the final objective-function value
          - ``rms``: the root-mean-square residual of the fit
          - ``solution``: the solution table as a list of record dicts
          - ``model``: the loaded DataModel (with fitted values applied when
            ``update_model`` is True), for re-simulation / verification.
    """
    dm = _as_datamodel(model_source)
    exp_df = _load_experiment(experiment)
    norm_params = normalize_fit_parameters(fit_params)

    # Register the reference trace as a fitting experiment. add_experiment maps
    # columns to model objects by name; the "Time" column is recognized as time.
    # Guard against a duplicate name if the same DataModel is reused across calls
    # (basico 0.86 has no remove_experiment, so fall back to a unique name).
    existing = basico.get_experiment_names(model=dm) or []
    if experiment_name in existing:
        remove = getattr(basico, "remove_experiment", None)
        if callable(remove):
            remove(experiment_name, model=dm)
        else:
            experiment_name = f"{experiment_name}_{len(existing)}"

    # add_experiment writes the DataFrame to "<name>.txt"; route it to a temp
    # directory so runs don't litter the working tree. The file only needs to
    # exist for the duration of the estimation run.
    data_dir = tempfile.mkdtemp(prefix="viva_copasi_pe_")
    try:
        basico.add_experiment(experiment_name, exp_df, model=dm, data_dir=data_dir)

        # Declare the fit items.
        basico.set_fit_parameters(norm_params, model=dm)

        # Run the optimizer.
        run_kwargs: Dict[str, Any] = {"update_model": update_model, "model": dm}
        if method is not None:
            run_kwargs["method"] = method
        solution = basico.run_parameter_estimation(**run_kwargs)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)

    # solution is a DataFrame indexed by fit-item name with a 'sol' column.
    fitted = {name: float(row["sol"]) for name, row in solution.iterrows()}

    stat = basico.get_fit_statistic(model=dm)
    objective = float(stat.get("obj")) if stat.get("obj") is not None else float("nan")
    rms = float(stat.get("rms")) if stat.get("rms") is not None else float("nan")

    solution_records = solution.reset_index().to_dict(orient="records")

    return {
        "fitted": fitted,
        "objective": objective,
        "rms": rms,
        "solution": solution_records,
        "model": dm,
    }


def resimulation_rmsd(
    model: Union[str, Any],
    reference: pd.DataFrame,
    *,
    fitted: Optional[Dict[str, float]] = None,
    duration: Optional[float] = None,
    intervals: Optional[int] = None,
    species: Optional[Sequence[str]] = None,
) -> float:
    """Re-simulate and return the RMSD of the trajectory vs a reference trace.

    Because ``basico.get_parameters`` read-back is unreliable, the trustworthy
    way to confirm a recovered fit is to re-run the simulation and compare the
    resulting trajectory against the reference. A value near zero means the
    (re-simulated) model reproduces the reference — i.e. the fit is good.

    Args:
        model: Path/URL to a model, or a loaded DataModel (e.g. the ``"model"``
            returned by :func:`estimate`, which already carries the fitted
            values when ``update_model`` was True).
        reference: The reference trace, with a ``"Time"`` column and species
            columns (as produced by :func:`build_experiment_dataframe`).
        fitted: Optional ``{name: value}`` overrides to apply before simulating
            (bare or ``Values[...]`` names accepted). Use this to verify a fit
            without relying on ``update_model``.
        duration: Simulation duration. Defaults to ``reference["Time"].max()``.
        intervals: Number of intervals. Defaults to ``len(reference) - 1`` so the
            re-simulated time points line up with the reference rows.
        species: Species columns to compare. Defaults to every non-Time column.

    Returns:
        The root-mean-square deviation between the re-simulated trajectory and
        the reference, pooled over all compared species and time points.
    """
    dm = _as_datamodel(model)

    if fitted:
        for name, value in fitted.items():
            bare = name
            if bare.startswith("Values[") and bare.endswith("]"):
                bare = bare[len("Values["):-1]
            basico.set_parameters(name=bare, exact=True, initial_value=float(value), model=dm)

    cols = list(species) if species is not None else [
        c for c in reference.columns if c != "Time"
    ]
    dur = float(duration) if duration is not None else float(reference["Time"].max())
    n_int = int(intervals) if intervals is not None else max(len(reference) - 1, 1)

    tc = basico.run_time_course(
        start_time=float(reference["Time"].min()),
        duration=dur,
        intervals=n_int,
        update_model=False,
        model=dm,
    )
    sim = build_experiment_dataframe(tc, species=cols)

    # Align on the shorter length in case of off-by-one row counts.
    n = min(len(sim), len(reference))
    diffs = []
    for c in cols:
        ref_vals = np.asarray(reference[c].to_numpy()[:n], dtype=float)
        sim_vals = np.asarray(sim[c].to_numpy()[:n], dtype=float)
        diffs.append(ref_vals - sim_vals)
    resid = np.concatenate(diffs) if diffs else np.array([0.0])
    return float(np.sqrt(np.mean(resid ** 2)))
