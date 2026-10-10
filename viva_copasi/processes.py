import warnings
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from pandas import DataFrame
from process_bigraph import Process, Step
import COPASI
from basico import (
    load_model,
    load_model_from_string,
    get_species,
    get_reactions,
    run_time_course,
    run_time_course_with_output,
    run_steadystate,
    get_value,
    get_parameters,
    set_parameters,
    set_species,
)

from viva_copasi.parameter_estimation import (
    estimate,
    build_experiment_dataframe,
    resimulation_rmsd,
)


# Species-not-found is a warning, not an error: partial updates are legitimate
# (a config may name species absent from a given model). warnings.warn (rather
# than print) surfaces through a workbench worker's warning capture instead of
# vanishing to stdout.


def _model_path_resolution(model_source: str) -> str:
    """Resolve a model reference to a loadable path or URL.

    URLs pass through unchanged; relative paths resolve against Path.cwd().
    Raises FileNotFoundError (naming the resolved path and cwd) if a file path
    does not exist, rather than letting load_model silently return None.
    """
    if model_source.startswith(('http://', 'https://')):
        return model_source
    p = Path(model_source)
    cwd = Path.cwd()
    if not p.is_absolute():
        p = cwd / p
    if not p.exists():
        raise FileNotFoundError(
            f"COPASI/SBML model not found: model_source={model_source!r} "
            f"resolved to {str(p)!r} (cwd={str(cwd)!r}). "
            f"Pass an absolute path or a path relative to the current "
            f"working directory ({cwd})."
        )
    return str(p)


def _looks_like_model_content(model_source: str) -> bool:
    """True when ``model_source`` is raw model TEXT (SBML or COPASI XML) rather
    than a filesystem path or URL (#15).

    COPASI and SBML documents are XML, so their first non-whitespace character
    is always ``<`` (e.g. ``<?xml``, ``<sbml``, ``<COPASI``). No filesystem path
    or http(s) URL begins that way, so a leading ``<`` unambiguously marks model
    content — including multiline SBML passed via ``open(...).read()``.
    """
    return isinstance(model_source, str) and model_source.lstrip().startswith('<')


def _load_model_source(model_source: str):
    """Load a COPASI/SBML model from either raw model text or a path/URL (#15).

    Raw model content (detected by a leading ``<``) is loaded with basico's
    string loader ``load_model_from_string``; anything else is resolved as a
    path/URL via :func:`_model_path_resolution` (preserving the prior behavior,
    including the FileNotFoundError on a missing path) and loaded from disk/URL.
    Returns the loaded ``COPASI.CDataModel``; raises if loading yields None.
    """
    if _looks_like_model_content(model_source):
        dm = load_model_from_string(model_source)
        if dm is None:
            raise RuntimeError(
                "load_model_from_string(...) returned None; the provided "
                "model_source text is not a valid COPASI/SBML model."
            )
        return dm
    resolved = _model_path_resolution(model_source)
    dm = load_model(resolved)
    if dm is None:
        raise RuntimeError(
            f"load_model({model_source!r}) returned None (resolved path: "
            f"{resolved!r}). "
            "Check that the file exists and is a valid COPASI/SBML model."
        )
    return dm


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
            warnings.warn(f"Species {name} not found in model; skipping")
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
        warnings.warn(f"Species {name} not found in model; returning None")
        return None
    assert isinstance(species, COPASI.CMetab)
    return float(species.getConcentration())


def _get_transient_amount(name, dm):
    """Return the *current* amount (particle number, in the model's quantity
    units) of a species. COPASI's particle number already folds in the
    compartment volume, so this is concentration * volume (times the quantity
    unit's Avogadro factor) — see issue #18."""
    model = dm.getModel()
    assert isinstance(model, COPASI.CModel)

    species = model.getMetabolite(name)
    if species is None:
        warnings.warn(f"Species {name} not found in model; returning None")
        return None
    assert isinstance(species, COPASI.CMetab)
    return float(species.getValue())


# Issue #18: COPASI reports a species' bare name ("S1") as particle count but
# "[S1]" as concentration, and which one its *default* output uses tracks the
# species' SBML hasOnlySubstanceUnits flag — so a model's default species output
# can silently mix concentrations and particle counts. The wrapper hides this:
# the `species_units` config key forces the default species output into one kind
# for every species. COPASI exposes both quantities directly (concentration vs
# particle-number reference), each already accounting for compartment volume, so
# the wrapper selects the reference explicitly rather than reading the per-species
# flag. Same key name / semantics as the viva-tellurium sibling (#14).
SPECIES_UNITS_CONCENTRATION = 'concentration'
SPECIES_UNITS_AMOUNT = 'amount'
_VALID_SPECIES_UNITS = (SPECIES_UNITS_CONCENTRATION, SPECIES_UNITS_AMOUNT)


class BaseCopasi:
    cmodel = None
    dm = None
    species_ids = None
    reaction_ids = None
    sbml_to_name = None

    def interpret_sbml(self):
        model_source = self.config['model_source']

        # ---- Load COPASI model ----
        # model_source may be a path/URL or raw SBML/COPASI model text (#15).
        self.dm = _load_model_source(model_source)

        self.cmodel = self.dm.getModel()

        # A valid ODE model may have NO species (only parameters + rate rules),
        # in which case basico's get_species returns None rather than an empty
        # frame (#27). Treat that as an empty species set and carry on instead
        # of indexing None (which raised TypeError: 'NoneType' is not
        # subscriptable).
        spec_df = get_species(model=self.dm)
        if spec_df is None:
            self.species_ids = []
            self.sbml_to_name = {}
        else:
            # External canonical IDs: SBML IDs
            self.species_ids = spec_df["sbml_id"].tolist()
            # Mapping: SBML ID -> COPASI display name (index)
            self.sbml_to_name = {
                spec_df.loc[name, "sbml_id"]: name
                for name in spec_df.index
            }

        # Likewise, a model with no reactions (e.g. the rate-rule-only model
        # above) yields None from get_reactions (#27).
        rxn_df = get_reactions(model=self.dm)
        # These are typically SBML reaction ids already
        self.reaction_ids = [] if rxn_df is None else rxn_df.index.tolist()

        # Snapshot the model's configured initial state so the zero-time Step
        # variants can reset to it before each firing (#28). Captured here, used
        # only by the Steps' reset — the stateful Process variant never resets.
        self._snapshot_initial_state()

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

    def _snapshot_initial_state(self) -> None:
        """Record the model's configured initial values right after loading, so
        a zero-time Step can restore them before each firing (#28).

        Captures the initial concentration of every species and the initial
        value of every global quantity (covers rate-rule-driven parameters, like
        the no-species model in #27). Reaction/compartment edge cases are out of
        scope; these two cover the stateful quantities COPASI advances in a time
        course."""
        sp = get_species(model=self.dm)
        self._initial_species = (
            {}
            if sp is None
            else {
                name: float(sp.loc[name, "initial_concentration"])
                for name in sp.index
            }
        )
        gq = get_parameters(model=self.dm)
        self._initial_globals = (
            {}
            if gq is None
            else {
                name: float(gq.loc[name, "initial_value"])
                for name in gq.index
            }
        )

    def _reset_to_initial_state(self) -> None:
        """Restore the initial values snapshotted in :meth:`_snapshot_initial_state`.

        Makes a zero-time Step a pure function of its inputs: without this, a
        prior ``run_time_course(update_model=True)`` leaves the model's initial
        state at the previous end-state and the next firing continues from there
        (#28). Called at the start of each Step ``update()``; the time-coupled
        Process variant deliberately does not call it."""
        for name, value in getattr(self, "_initial_species", {}).items():
            set_species(name=name, exact=True, initial_concentration=value, model=self.dm)
        for name, value in getattr(self, "_initial_globals", {}).items():
            set_parameters(name=name, exact=True, initial_value=value, model=self.dm)

    def species_units(self) -> str:
        """The default-species-output unit policy (#18): 'concentration'
        (default) or 'amount'. Validated so a typo fails loudly rather than
        silently falling back."""
        units = self.config.get('species_units') or SPECIES_UNITS_CONCENTRATION
        if units not in _VALID_SPECIES_UNITS:
            raise ValueError(
                f"species_units={units!r} is invalid; expected one of "
                f"{_VALID_SPECIES_UNITS}."
            )
        return units

    def timecourse_option_kwargs(self) -> Dict[str, Any]:
        """Translate the simulation-option config keys into basico
        ``run_time_course`` kwargs.

        Keys left unset (None) are omitted so basico applies its own defaults,
        preserving the previous behavior when no options are supplied.

        Key vocabulary (shared with the tellurium wrapper where the concept
        exists):

        * ``method``             -> ``method``  (basico method name, e.g.
          ``deterministic``/``lsoda``, ``stochastic``, ``directMethod`` ...).
          Tellurium calls the analogous key ``integrator`` (``cvode``/
          ``gillespie``); COPASI's native term is ``method``, matching basico
          and the existing ``ParameterEstimationStep``.
        * ``relative_tolerance`` -> ``r_tol``   (matches tellurium's key name)
        * ``absolute_tolerance`` -> ``a_tol``   (matches tellurium's key name)
        * ``step_size``          -> ``stepsize`` (output step size)
        """
        kw: Dict[str, Any] = {}
        method = self.config.get('method')
        if method:
            kw['method'] = method
        r_tol = self.config.get('relative_tolerance')
        if r_tol is not None:
            kw['r_tol'] = float(r_tol)
        a_tol = self.config.get('absolute_tolerance')
        if a_tol is not None:
            kw['a_tol'] = float(a_tol)
        step_size = self.config.get('step_size')
        if step_size is not None:
            kw['stepsize'] = float(step_size)
        return kw


class CopasiUTCStep(Step, BaseCopasi):

    config_schema = {
        'model_source': 'string',
        'time': 'float',
        'n_points': 'integer',
        # Output start time; default 0.0 preserves prior behavior (#13).
        'start_time': {'_type': 'float', '_default': 0.0},
        # Explicit output time points (#16): when set, the time course reports
        # output at exactly these times (via basico's `values=` argument),
        # overriding the uniform start_time/time/n_points path. Absent (None)
        # keeps the current uniform behavior. Same key name and list shape as the
        # viva-tellurium sibling (#11).
        'output_times': 'maybe[list[float]]',
        # Simulation options; absent (None) -> basico defaults (#14).
        'method': 'maybe[string]',
        'relative_tolerance': 'maybe[float]',
        'absolute_tolerance': 'maybe[float]',
        'step_size': 'maybe[float]',
        # Output selection (#17): list of element identifiers to report, e.g.
        # ['Time', '[LacI protein]', '(reaction).Flux']. Absent/empty keeps the
        # current behavior (basico's default species+parameter output). Same key
        # name and list shape as the tellurium wrapper's `selections` (#13); the
        # identifier vocabulary is COPASI's (display names / CNs), analogous to
        # how `method` uses COPASI's vocabulary while the key name is shared.
        'selections': 'maybe[list[string]]',
        # Units for the default species output (#18): 'concentration' (default,
        # prior behavior) or 'amount' (particle number). Ignored when explicit
        # `selections` are given — those are reported verbatim. Same key name /
        # semantics as the viva-tellurium sibling (#14).
        'species_units': {'_type': 'string', '_default': SPECIES_UNITS_CONCENTRATION},
    }

    def initialize(self, config=None):
        self.interpret_sbml()

        # Simulation parameters
        self.interval = float(self.config.get('time', 1.0))
        self.n_points = int(self.config.get('n_points', 2))   # <-- NEW

        # Explicit output time points (#16). When given, they drive output and
        # the uniform n_points requirement does not apply.
        self.output_times = list(self.config.get('output_times') or [])

        if not self.output_times and self.n_points < 2:
            raise ValueError("n_points must be >= 2")

        self.intervals = self.n_points - 1   # COPASI requires this

        # Output start time (#13): honor config, default 0.0.
        self.start_time = float(self.config.get('start_time') or 0.0)

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
        # A zero-time Step is a pure function of its inputs: reset the model to
        # its configured initial state so each firing starts fresh, rather than
        # continuing from the previous run's end-state (#28).
        self._reset_to_initial_state()

        # Apply incoming concentrations
        spec_data = inputs.get('species_counts', {}) or {}
        changes = [
            (name, float(value))
            for name, value in spec_data.items()
            if name in self.species_ids
        ]

        if changes:
            _set_initial_concentrations(changes, self.dm)

        selections = self.config.get('selections') or None

        # Time-span kwargs (#16): explicit `output_times` collect output at
        # exactly those points via basico's `values=`, overriding the uniform
        # start_time/duration/intervals path. Absent -> the uniform path,
        # unchanged.
        if self.output_times:
            time_kwargs = {'values': [float(t) for t in self.output_times]}
        else:
            time_kwargs = {
                'start_time': self.start_time,
                'duration': self.config['time'],
                'intervals': self.intervals,
            }

        if selections:
            # Output selection (#17): report exactly the requested elements via
            # basico's output_selection path. The selection strings are COPASI
            # display names / CNs (e.g. 'Time', '[species]', '(reaction).Flux').
            # run_time_course_with_output returns 'Time' as a regular column
            # (not the index) and does not take use_sbml_id.
            tc: DataFrame = run_time_course_with_output(
                output_selection=list(selections),
                update_model=True,
                model=self.dm,
                **time_kwargs,
                **self.timecourse_option_kwargs(),
            )
            if 'Time' in tc.columns:
                time_list = tc['Time'].to_list()
                columns = [c for c in tc.columns if c != 'Time']
            else:
                time_list = tc.index.to_list()
                columns = [c for c in tc.columns]
            result = {
                "time": time_list,
                "columns": columns,
                "values": tc[columns].values.tolist(),
            }
            return {"result": result}

        # --- Run COPASI time course with intervals = n_points - 1 ---
        # species_units (#18): concentration (basico default) vs amount
        # (particle number). use_concentrations=False makes basico return the
        # particle-number data for species columns; global-quantity columns are
        # unaffected (their value is the same either way).
        tc: DataFrame = run_time_course(
            update_model=True,
            use_sbml_id=True,
            use_concentrations=(self.species_units() == SPECIES_UNITS_CONCENTRATION),
            model=self.dm,
            **time_kwargs,
            **self.timecourse_option_kwargs(),
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
        # Steady-state options; absent (None) -> basico/COPASI defaults (#14).
        # relative_tolerance maps to the Enhanced Newton method's acceptance
        # 'Resolution'; criterion is the acceptance 'Target Criterion'
        # ('Distance and Rate' | 'Distance' | 'Rate'). The time-course-only
        # options (method name, absolute_tolerance, step_size) do not apply to
        # the steady-state task.
        'relative_tolerance': 'maybe[float]',
        'criterion': 'maybe[string]',
        # Remaining Steady-State task options (#19). Each maps to one setting of
        # the COPASI "Enhanced Newton" steady-state method (names as returned by
        # basico.get_task_settings(T.STEADY_STATE)['method']). Absent (None) ->
        # COPASI's default for that setting, so the default run is unchanged.
        # These names are engine-specific (Enhanced Newton); the tellurium
        # sibling (#15) exposes its own roadrunner steady-state options.
        'derivation_factor': 'maybe[float]',               # 'Derivation Factor'
        'iteration_limit': 'maybe[integer]',               # 'Iteration Limit'
        'use_newton': 'maybe[boolean]',                    # 'Use Newton'
        'use_integration': 'maybe[boolean]',               # 'Use Integration'
        'use_back_integration': 'maybe[boolean]',          # 'Use Back Integration'
        'accept_negative_concentrations': 'maybe[boolean]',  # 'Accept Negative Concentrations'
        'forward_integration_duration': 'maybe[float]',    # 'Maximum duration for forward integration'
        'backward_integration_duration': 'maybe[float]',   # 'Maximum duration for backward integration'
        # Output selection (#17): list of element identifiers whose steady-state
        # values to report, e.g. ['[LacI protein]', '(reaction).Flux']. Absent/
        # empty keeps the current behavior. Same key name and list shape as the
        # tellurium wrapper's `selections` (#13); the identifiers are COPASI
        # display names / CNs. basico has no steady-state output_selection, so
        # each element's steady-state value is read back via basico.get_value.
        'selections': 'maybe[list[string]]',
        # Units for the default species output (#18): 'concentration' (default,
        # prior behavior) or 'amount' (particle number). Applies to the
        # `species_concentrations` values; explicit `selections` are verbatim.
        'species_units': {'_type': 'string', '_default': SPECIES_UNITS_CONCENTRATION},
    }

    def initialize(self, config=None):
        self.interpret_sbml()

    # (config key, Enhanced-Newton method setting name, cast) for the
    # steady-state options exposed in #19. Resolution and Target Criterion are
    # the pre-existing relative_tolerance/criterion keys (#14) and handled below.
    _SS_METHOD_OPTIONS = (
        ('derivation_factor', 'Derivation Factor', float),
        ('iteration_limit', 'Iteration Limit', int),
        ('use_newton', 'Use Newton', bool),
        ('use_integration', 'Use Integration', bool),
        ('use_back_integration', 'Use Back Integration', bool),
        ('accept_negative_concentrations', 'Accept Negative Concentrations', bool),
        ('forward_integration_duration',
         'Maximum duration for forward integration', float),
        ('backward_integration_duration',
         'Maximum duration for backward integration', float),
    )

    def steadystate_option_kwargs(self) -> Dict[str, Any]:
        """Translate steady-state config keys into basico ``run_steadystate``
        kwargs. Unset (None) keys are omitted so COPASI's defaults apply."""
        kw: Dict[str, Any] = {}
        method: Dict[str, Any] = {}
        r_tol = self.config.get('relative_tolerance')
        if r_tol is not None:
            # Enhanced Newton acceptance resolution, in get_task_settings form.
            method['Resolution'] = float(r_tol)
        # Remaining Enhanced-Newton method options (#19).
        for cfg_key, setting_name, cast in self._SS_METHOD_OPTIONS:
            val = self.config.get(cfg_key)
            if val is not None:
                method[setting_name] = cast(val)
        if method:
            kw['settings'] = {'method': method}
        criterion = self.config.get('criterion')
        if criterion:
            kw['criterion'] = criterion
        return kw

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
        # Zero-time Step: reset to the configured initial state so repeated
        # firings are a pure function of the inputs (#28). run_steadystate with
        # update_model=True otherwise leaves the model at the solved state, so a
        # prior solve could seed the next one.
        self._reset_to_initial_state()

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
            **self.steadystate_option_kwargs(),
        )

        # 3) Read back steady-state species values (SBML IDs externally).
        # species_units (#18): 'concentration' -> the 'concentration' column;
        # 'amount' -> the 'particle_number' column (amount, volume-aware).
        spec_df = get_species(model=self.dm)
        # spec_df is indexed by COPASI name, with 'sbml_id', 'concentration'
        # and 'particle_number' columns.
        value_col = (
            "concentration"
            if self.species_units() == SPECIES_UNITS_CONCENTRATION
            else "particle_number"
        )
        species_conc_ss = {}
        for name in spec_df.index:
            sbml_id = spec_df.loc[name, "sbml_id"]
            if sbml_id in self.species_ids:
                species_conc_ss[sbml_id] = float(spec_df.loc[name, value_col])

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

        # Output selection (#17): report the steady-state value of each
        # requested element. Keyed by the selection string, each value is a
        # one-element list to match the one-point "time series" convention used
        # by species_concentrations / fluxes above. Only added when requested,
        # so the default output is unchanged.
        selections = self.config.get('selections') or None
        if selections:
            selection_json = {}
            for sel in selections:
                v = get_value(sel, model=self.dm)
                selection_json[sel] = [float(v)] if v is not None else [None]
            results["selections"] = selection_json

        return {"results": results}


class CopasiUTCProcess(Process, BaseCopasi):

    config_schema = {
        'model_source': 'string',
        'time': 'float',
        'intervals': 'integer',
        # Output start time; default 0.0 preserves prior behavior (#13).
        'start_time': {'_type': 'float', '_default': 0.0},
        # Simulation options; absent (None) -> basico defaults (#14).
        'method': 'maybe[string]',
        'relative_tolerance': 'maybe[float]',
        'absolute_tolerance': 'maybe[float]',
        'step_size': 'maybe[float]',
        # Units for the species_concentrations output (#18): 'concentration'
        # (default, prior behavior) or 'amount' (particle number). Same key name
        # / semantics as the viva-tellurium sibling (#14).
        'species_units': {'_type': 'string', '_default': SPECIES_UNITS_CONCENTRATION},
    }

    def initialize(self, config=None):
        self.interpret_sbml()

        # ---- Sim parameters ----
        self.time = float(self.config.get("time", 1.0))
        self.intervals = int(self.config.get("intervals", 10))

        # Output start time (#13): honor config, default 0.0.
        self.start_time = float(self.config.get('start_time') or 0.0)

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
            start_time=self.start_time,
            duration=interval,
            intervals=self.intervals,
            update_model=True,
            use_sbml_id=True,   # <-- critical
            model=self.dm,
            **self.timecourse_option_kwargs(),
        )

        # Extract time points
        time = tc.index.tolist()

        # --- 3) Read back final state: export SBML IDs ----
        # species_units (#18): concentration (default) vs amount (particle no.).
        read = (
            _get_transient_concentration
            if self.species_units() == SPECIES_UNITS_CONCENTRATION
            else _get_transient_amount
        )
        species_concentrations = {
            sbml_id: read(
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
            # Accepts a path/URL or raw model text, like model_source (#15).
            ref_dm = _load_model_source(self.reference_model_source)
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
