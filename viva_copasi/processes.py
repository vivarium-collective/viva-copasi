from pathlib import Path
from typing import Dict, Any

import pandas as pd
from pandas import DataFrame
from process_bigraph import Process, Step
import COPASI
from basico import (
    load_model,
    get_species,
    get_reactions,
    run_time_course,
    run_steadystate,
    set_parameters,
)

from viva_copasi.parameter_estimation import (
    estimate,
    build_experiment_dataframe,
    resimulation_rmsd,
)


def _model_path_resolution(model_source: str) -> str:
    """Resolve a model reference to a loadable path or URL.

    URLs pass through unchanged; relative paths resolve against Path.cwd().
    """
    if model_source.startswith(('http://', 'https://')):
        return model_source
    p = Path(model_source)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def _set_initial_concentrations(changes, dm):
    """
    changes: iterable of (species_name, value) pairs
    dm: COPASI DataModel as returned by basico.load_model
    """
    model = dm.getModel()
    assert isinstance(model, COPASI.CModel)

    references = COPASI.ObjectStdVector()

    for name, value in changes:
        species = model.getMetabolite(name)
        if species is None:
            print(f"Species {name} not found in model")
            continue
        assert isinstance(species, COPASI.CMetab)
        species.setInitialConcentration(float(value))
        references.append(species.getInitialConcentrationReference())

    if len(references) > 0:
        model.updateInitialValues(references)


def _get_transient_concentration(name, dm):
    """
    Return the *current* concentration (not initial) of a species.
    """
    model = dm.getModel()
    assert isinstance(model, COPASI.CModel)

    species = model.getMetabolite(name)
    if species is None:
        print(f"Species {name} not found in model")
        return None
    assert isinstance(species, COPASI.CMetab)
    return float(species.getConcentration())


class BaseCopasi:
    cmodel = None
    dm = None
    species_ids = None
    reaction_ids = None
    sbml_to_name = None

    def interpret_sbml(self):
        model_source = self.config['model_source']

        # ---- Load COPASI model ----
        self.dm = load_model(_model_path_resolution(model_source))
        if self.dm is None:
            raise RuntimeError(
                f"load_model({model_source!r}) returned None. "
                "Check that the file exists and is a valid COPASI/SBML model."
            )

        self.cmodel = self.dm.getModel()

        spec_df = get_species(model=self.dm)

        # External canonical IDs: SBML IDs
        self.species_ids = spec_df["sbml_id"].tolist()

        # Mapping: SBML ID -> COPASI display name (index)
        self.sbml_to_name = {
            spec_df.loc[name, "sbml_id"]: name
            for name in spec_df.index
        }

        rxn_df = get_reactions(model=self.dm)
        # These are typically SBML reaction ids already
        self.reaction_ids = rxn_df.index.tolist()

    def get_concentrations_from_sbml(self) -> Dict[str, Any]:
        return {
            "species_concentrations": {
                sbml_id: _get_transient_concentration(
                    name=self.sbml_to_name[sbml_id],  # COPASI name
                    dm=self.dm
                )
                for sbml_id in self.species_ids
            }
        }


class CopasiUTCStep(Step, BaseCopasi):

    config_schema = {
        'model_source': 'string',
        'time': 'float',
        'n_points': 'integer',
    }

    def initialize(self, config=None):
        self.interpret_sbml()

        # Simulation parameters
        self.interval = float(self.config.get('time', 1.0))
        self.n_points = int(self.config.get('n_points', 2))   # <-- NEW
        if self.n_points < 2:
            raise ValueError("n_points must be >= 2")

        self.intervals = self.n_points - 1   # COPASI requires this

    def initial_state(self) -> Dict[str, Any]:
        return self.get_concentrations_from_sbml()

    def inputs(self):
        return {
            'species_concentrations': 'map[float]',
            'species_counts': 'map[float]',
        }

    def outputs(self):
        return {
            'result': 'numeric_result',
        }

    def update(self, inputs):
        # Apply incoming concentrations
        spec_data = inputs.get('species_counts', {}) or {}
        changes = [
            (name, float(value))
            for name, value in spec_data.items()
            if name in self.species_ids
        ]

        if changes:
            _set_initial_concentrations(changes, self.dm)

        # --- Run COPASI time course with intervals = n_points - 1 ---
        tc: DataFrame = run_time_course(
            start_time=0.0,
            duration=self.config['time'],
            intervals=self.intervals,
            update_model=True,
            use_sbml_id=True,
            model=self.dm,
        )

        # Time series
        time_list = tc.index.to_list()

        result = {
            "time": time_list,
            "columns": [c for c in tc.columns],
            "values": tc.values.tolist(),
            # "n_spacial_dimensions": tc.shape,
        }

        return {"result": result}


class CopasiSteadyStateStep(Step, BaseCopasi):

    config_schema = {
        'model_source': 'string',
        'time': 'float',  # kept for symmetry, not used
    }

    def initialize(self, config=None):
        self.interpret_sbml()

    # ------------------------------------------------
    # initial state (SBML IDs externally)
    # ------------------------------------------------
    def initial_state(self) -> Dict[str, Any]:
        return self.get_concentrations_from_sbml()

    # ------------------------------------------------
    # ports
    # ------------------------------------------------
    def inputs(self):
        # Externally everything uses SBML IDs
        return {
            'species_concentrations': 'map[float]',  # SBML IDs
            'counts': 'map[float]',          # SBML IDs
        }

    def outputs(self):
        # Match TelluriumSteadyStateStep: nested results
        return {
            'results': 'any',
        }

    # ------------------------------------------------
    # steady-state update
    # ------------------------------------------------
    def update(self, inputs):
        # 1) Prefer counts, otherwise concentrations (keys are SBML IDs)
        spec_data = (
            inputs.get('counts')
            or inputs.get('concentrations')
            or {}
        )

        # Convert SBML IDs -> COPASI names for internal set
        changes = []
        for sbml_id, value in spec_data.items():
            name = self.sbml_to_name.get(sbml_id)
            if name is not None:
                changes.append((name, float(value)))

        if changes:
            _set_initial_concentrations(changes, self.dm)

        # 2) Run COPASI steady-state task
        # (use_sbml_id affects task I/O naming, but we read from get_species anyway)
        run_steadystate(
            update_model=True,
            use_sbml_id=True,
            model=self.dm,
        )

        # 3) Read back steady-state species concentrations (SBML IDs externally)
        spec_df = get_species(model=self.dm)
        # spec_df is indexed by COPASI name, with 'sbml_id' and 'concentration' columns
        species_conc_ss = {}
        for name in spec_df.index:
            sbml_id = spec_df.loc[name, "sbml_id"]
            if sbml_id in self.species_ids:
                species_conc_ss[sbml_id] = float(spec_df.loc[name, "concentration"])

        # 4) Steady-state reaction fluxes
        rxn_df = get_reactions(model=self.dm)
        reaction_fluxes_ss = {
            rid: float(rxn_df.loc[rid, 'flux'])
            for rid in self.reaction_ids
            if rid in rxn_df.index
        }

        # 5) Package as one-point "time series" (t = 0.0) to match Tellurium
        time_list = [0.0]

        species_json = {sid: [val] for sid, val in species_conc_ss.items()}
        flux_json = {rid: [val] for rid, val in reaction_fluxes_ss.items()}

        results = {
            "time": time_list,
            "species_concentrations": species_json,  # SBML IDs as keys
            "fluxes": flux_json,
        }

        return {"results": results}


class CopasiUTCProcess(Process, BaseCopasi):

    config_schema = {
        'model_source': 'string',
        'time': 'float',
        'intervals': 'integer',
    }

    def initialize(self, config=None):
        self.interpret_sbml()

        # ---- Sim parameters ----
        self.time = float(self.config.get("time", 1.0))
        self.intervals = int(self.config.get("intervals", 10))

    # -----------------------------------------------------------------
    # initial state
    # -----------------------------------------------------------------
    def initial_state(self) -> Dict[str, Any]:
        return self.get_concentrations_from_sbml()

    # -----------------------------------------------------------------
    # I/O schema
    # -----------------------------------------------------------------
    def inputs(self):
        return {
            "species_concentrations": "map[float]",  # SBML IDs
            "species_counts": "map[float]",          # SBML IDs
        }

    def outputs(self):
        return {
            "species_concentrations": "map[float]",  # SBML IDs
            "fluxes": "map[float]",
            "time": "list[float]",
        }

    # -----------------------------------------------------------------
    # update
    # -----------------------------------------------------------------
    def update(self, inputs, interval):
        # --- 1) Determine incoming species map (SBML IDs)
        incoming = (
            inputs.get("species_counts")
            or inputs.get("species_concentrations")
            or {}
        )

        # Convert SBML IDs -> COPASI names for internal setting
        changes = []
        for sbml_id, value in incoming.items():
            name = self.sbml_to_name.get(sbml_id)
            if name is not None:
                changes.append((name, float(value)))

        if changes:
            _set_initial_concentrations(changes, self.dm)

        # --- 2) Run time course with SBML-ID columns ----
        tc = run_time_course(
            start_time=0.0,
            duration=interval,
            intervals=self.intervals,
            update_model=True,
            use_sbml_id=True,   # <-- critical
            model=self.dm,
        )

        # Extract time points
        time = tc.index.tolist()

        # --- 3) Read back final state: export SBML IDs ----
        species_concentrations = {
            sbml_id: _get_transient_concentration(
                name=self.sbml_to_name[sbml_id],
                dm=self.dm
            )
            for sbml_id in self.species_ids
        }

        # --- 4) Reaction fluxes (COPASI reaction IDs already match SBML IDs) ----
        rxn_df = get_reactions(model=self.dm)
        reaction_fluxes = {
            rxn_id: float(rxn_df.loc[rxn_id, "flux"])
            for rxn_id in self.reaction_ids
        }

        return {
            "species_concentrations": species_concentrations,
            "fluxes": reaction_fluxes,
            "time": time,
        }


class ParameterEstimationStep(Step, BaseCopasi):
    """One-shot parameter estimation via COPASI's Parameter Estimation task.

    Fits ``fit_params`` of ``model_source`` so a simulated time course matches a
    reference trace. The reference is supplied either inline (``experiment_data``)
    or generated by simulating ``reference_model_source`` (defaulting to
    ``model_source``). ``param_overrides`` optionally perturbs the fit model
    *after* the reference is generated (e.g. to break a known parameter for a
    recovery test). Wraps :func:`viva_copasi.parameter_estimation.estimate`.

    Being a fit (not a time-advancing process) it is modeled as a Step: it runs
    once and emits the fitted values plus fit statistics.
    """

    # Only simply-typed keys are declared here; bigraph-schema's fill() has no
    # default for 'any', so the structured inputs (fit_params, experiment_data,
    # reference_species) are passed through config untyped and read in
    # initialize() — fill() preserves undeclared config keys.
    config_schema = {
        'model_source': 'string',
        'reference_model_source': 'string',
        'reference_duration': 'float',
        'reference_intervals': 'integer',
        'param_overrides': 'map[float]',  # perturb fit model before fitting
        'method': 'string',
    }

    def initialize(self, config=None):
        self.interpret_sbml()

        self.fit_params = list(self.config.get('fit_params') or [])
        if not self.fit_params:
            raise ValueError("ParameterEstimationStep requires non-empty 'fit_params'.")

        self.reference_model_source = self.config.get('reference_model_source') or None
        self.reference_species = self.config.get('reference_species') or None
        self.reference_duration = float(self.config.get('reference_duration', 100.0))
        self.reference_intervals = int(self.config.get('reference_intervals', 100))
        self.param_overrides = dict(self.config.get('param_overrides') or {})
        self.method = self.config.get('method') or None
        self.update_model = bool(self.config.get('update_model', True))

        # Keep the reference trace around for verify-by-resimulation.
        self._reference_df = None

    def _coerce_experiment(self, data) -> DataFrame:
        """Coerce inline experiment_data into a DataFrame with a Time column."""
        if isinstance(data, DataFrame):
            return data
        # list of record dicts, or dict of columns
        return pd.DataFrame(data)

    def _build_reference(self) -> DataFrame:
        """Return the reference trace, from inline data or by simulation."""
        inline = self.config.get('experiment_data')
        if inline is not None:
            return self._coerce_experiment(inline)

        # Simulate the reference. Use a separately loaded model when a distinct
        # reference source is given, else simulate from the (as-yet-unperturbed)
        # fit model so its file/default parameters act as ground truth.
        if self.reference_model_source is not None:
            ref_dm = load_model(_model_path_resolution(self.reference_model_source))
        else:
            ref_dm = self.dm
        tc = run_time_course(
            duration=self.reference_duration,
            intervals=self.reference_intervals,
            update_model=False,
            model=ref_dm,
        )
        return build_experiment_dataframe(tc, species=self.reference_species)

    def inputs(self):
        # Self-contained fit; no upstream inputs required.
        return {}

    def outputs(self):
        return {
            'fitted': 'map[float]',
            'objective': 'float',
            'rms': 'float',
            'resim_rmsd': 'float',
            'solution': 'any',
        }

    def update(self, inputs):
        # 1) Obtain the reference trace (before perturbing the fit model).
        reference = self._build_reference()
        self._reference_df = reference

        # 2) Optionally perturb ("break") the fit model's parameters.
        for name, value in self.param_overrides.items():
            set_parameters(name=name, exact=True, initial_value=float(value), model=self.dm)

        # 3) Run parameter estimation against the fit model.
        result = estimate(
            self.dm,
            reference,
            self.fit_params,
            method=self.method,
            update_model=self.update_model,
        )

        # 4) Verify by re-simulation (get_parameters read-back is unreliable).
        species = self.reference_species or [
            c for c in reference.columns if c != "Time"
        ]
        resim = resimulation_rmsd(result["model"], reference, species=species)

        return {
            'fitted': result['fitted'],
            'objective': result['objective'],
            'rms': result['rms'],
            'resim_rmsd': resim,
            'solution': result['solution'],
        }
