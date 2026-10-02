"""COPASI composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — ``make_copasi_utc_document(model_source=..., ...)``
   builds a PBG state-dict programmatically for callers that want full control
   over the model path + wiring.

2. **Declarative ``*.composite.yaml``** — sibling files in this directory follow
   the pbg-superpowers composite-spec convention.  ``build_composite()`` loads
   one by name and instantiates ``process_bigraph.Composite`` with parameter
   substitution.  The dashboard's composite explorer discovers these
   automatically once the package is installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from viva_copasi.processes import (
    CopasiUTCStep,
    CopasiSteadyStateStep,
    CopasiUTCProcess,
)


# ---------------------------------------------------------------------------
# Hand-coded composite factories (programmatic API)
# ---------------------------------------------------------------------------

def make_copasi_utc_document(
    model_source: str = '',
    time: float = 10.0,
    n_points: int = 11,
):
    """Create a composite document for a COPASI UTC step simulation.

    Returns a document dict ready for use with ``Composite()``.

    Args:
        model_source: Path or URL to an SBML/COPASI model file.
        time:         Simulation end time (model time units).
        n_points:     Number of output points (intervals = n_points - 1).

    Returns:
        dict: Composite document with CopasiUTCStep, stores, and emitter.
    """
    return {
        'copasi_utc': {
            '_type': 'step',
            'address': 'local:CopasiUTCStep',
            'config': {
                'model_source': model_source,
                'time': time,
                'n_points': n_points,
            },
            'inputs': {
                'species_concentrations': ['stores', 'species_concentrations'],
                'species_counts': ['stores', 'species_counts'],
            },
            'outputs': {
                'result': ['stores', 'result'],
            },
        },
        'stores': {
            'species_concentrations': {},
            'species_counts': {},
            'result': {},
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:RAMEmitter',
            'config': {
                'emit': {},
            },
            'inputs': {
                'result': ['stores', 'result'],
            },
        },
    }


def make_copasi_utc_process_document(
    model_source: str = '',
    time: float = 1.0,
    intervals: int = 10,
):
    """Create a composite document for a COPASI UTC incremental process.

    Args:
        model_source: Path or URL to an SBML/COPASI model file.
        time:         Duration per process update (model time units).
        intervals:    Number of internal integration intervals per update.

    Returns:
        dict: Composite document with CopasiUTCProcess, stores, and emitter.
    """
    return {
        'copasi_utc_process': {
            '_type': 'process',
            'address': 'local:CopasiUTCProcess',
            'config': {
                'model_source': model_source,
                'time': time,
                'intervals': intervals,
            },
            'inputs': {
                'species_concentrations': ['stores', 'species_concentrations'],
                'species_counts': ['stores', 'species_counts'],
            },
            'outputs': {
                'species_concentrations': ['stores', 'species_concentrations'],
                'fluxes': ['stores', 'fluxes'],
                'time': ['stores', 'time'],
            },
            'interval': time,
        },
        'stores': {
            'species_concentrations': {},
            'species_counts': {},
            'fluxes': {},
            'time': [],
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:RAMEmitter',
            'config': {
                'emit': {
                    'species_concentrations': 'map[float]',
                    'fluxes': 'map[float]',
                },
            },
            'inputs': {
                'species_concentrations': ['stores', 'species_concentrations'],
                'fluxes': ['stores', 'fluxes'],
            },
        },
    }


def make_copasi_steady_state_document(
    model_source: str = '',
):
    """Create a composite document for a COPASI steady-state solve.

    Args:
        model_source: Path or URL to an SBML/COPASI model file.

    Returns:
        dict: Composite document with CopasiSteadyStateStep, stores, and emitter.
    """
    return {
        'copasi_ss': {
            '_type': 'step',
            'address': 'local:CopasiSteadyStateStep',
            'config': {
                'model_source': model_source,
            },
            'inputs': {
                'species_concentrations': ['ss_stores', 'species_concentrations'],
                'counts': ['ss_stores', 'counts'],
            },
            'outputs': {
                'results': ['ss_stores', 'results'],
            },
        },
        'ss_stores': {
            'species_concentrations': {},
            'counts': {},
            'results': {},
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:RAMEmitter',
            'config': {
                'emit': {},
            },
            'inputs': {
                'results': ['ss_stores', 'results'],
            },
        },
    }


def register_copasi(core=None):
    """Return a core with CopasiUTCStep, CopasiSteadyStateStep, CopasiUTCProcess,
    the RAM emitter, and the SpeciesConcentrationsPlot visualization registered.

    Also registers 'any' and 'numeric_result' as tree-based schema types so
    composite wiring works for processes that use these output types.
    """
    if core is None:
        core = allocate_core()
    # Register schema types used by COPASI process outputs.
    # 'any' and 'numeric_result' are not in bigraph-schema's BASE_TYPES;
    # registering them as tree aliases allows composite wiring to resolve them.
    if 'any' not in core.registry:
        core.register_type('any', {'_inherit': 'tree'})
    if 'numeric_result' not in core.registry:
        core.register_type('numeric_result', {'_inherit': 'tree'})
    # Register process and emitter links.
    core.register_link('CopasiUTCStep', CopasiUTCStep)
    core.register_link('CopasiSteadyStateStep', CopasiSteadyStateStep)
    core.register_link('CopasiUTCProcess', CopasiUTCProcess)
    core.register_link('RAMEmitter', RAMEmitter)
    core.register_link('ram-emitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    from viva_copasi.visualizations import SpeciesConcentrationsPlot
    core.register_link('SpeciesConcentrationsPlot', SpeciesConcentrationsPlot)
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if value is None:
        return value
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _resolve_str(s: str, params: dict, overrides: dict) -> str:
    """Expand all ${VAR} occurrences in s using overrides then param defaults."""
    def _lookup(mm):
        pname = mm.group(1)
        val = overrides.get(pname, params.get(pname, {}).get("default", ""))
        # Recursively resolve nested placeholders in the looked-up value.
        if isinstance(val, str) and _INLINE_PLACEHOLDER.search(val):
            val = _INLINE_PLACEHOLDER.sub(_lookup, val)
        return str(val)
    return _INLINE_PLACEHOLDER.sub(_lookup, s)


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            # If the raw default itself contains placeholders, resolve them.
            if isinstance(raw, str) and _INLINE_PLACEHOLDER.search(raw):
                raw = _resolve_str(raw, params, overrides)
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _resolve_str(state, params, overrides)
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every ``*.composite.yaml`` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec.  ``name`` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a ``*.composite.yaml`` by name and instantiate process_bigraph.Composite.

    Args:
        name:      Short name of the composite spec (no ``.composite.yaml`` suffix).
        overrides: Parameter overrides (keys must match spec.parameters).
        core:      Optional pre-built core; otherwise ``register_copasi()`` is used.

    Returns:
        process_bigraph.Composite instance.

    Notes:
        ``model_dir`` is automatically injected into overrides as the absolute
        path to the composites package directory, so YAML defaults like
        ``${model_dir}/repressilator.xml`` resolve to the bundled SBML file.
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_copasi()

    # Auto-inject model_dir so yaml defaults like "${model_dir}/repressilator.xml"
    # resolve to the bundled model shipped alongside the yaml files.
    resolved_overrides = {'model_dir': str(_COMPOSITES_DIR)}
    resolved_overrides.update(overrides or {})

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, resolved_overrides)
    return Composite({"state": state}, core=core)
