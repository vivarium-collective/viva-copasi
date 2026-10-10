# viva-copasi

process-bigraph-compatible COPASI Steps and Processes for SBML model simulation.

Exposes any SBML kinetic model as a `process-bigraph` Step or Process so it can
be composed with other simulators in a bigraph document, using
[COPASI](https://copasi.org/) / [basico](https://basico.readthedocs.io/) as the
simulation back-end.

## Installation

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```python
from importlib.resources import files
from process_bigraph import Composite, gather_emitter_results
from viva_copasi.composites import build_composite, register_copasi

# Resolve the bundled Repressilator model
model_path = str(files('viva_copasi.composites').joinpath('repressilator.xml'))

# Build and run the incremental UTC process composite
sim = build_composite('utc-process')
for _ in range(10):
    sim.update({}, 1.0)   # advance 1 time unit per step

results = gather_emitter_results(sim)
print(results[('emitter',)][-1])
```

## Classes

- `CopasiUTCStep` — one-shot Uniform Time Course (UTC) trajectory (Step)
- `CopasiUTCProcess` — incremental UTC simulation (Process)
- `CopasiSteadyStateStep` — steady-state solve (Step)
- `BaseCopasi` — shared model-load mixin (SBML parsing, species/reaction indexing)

All classes accept a `model_source` config key pointing to an SBML or COPASI
model file (path or URL).

## Composites

Three discoverable composites in `viva_copasi/composites/`:

| Name | Type | Description |
|---|---|---|
| `utc-step` | Step | One-shot UTC trajectory; returns full time series in a single `update()` call |
| `utc-process` | Process | Incremental UTC; advances one interval per `update()`, emitting concentrations + fluxes |
| `steady-state` | Step | Steady-state solve; returns equilibrium concentrations and reaction fluxes |

Each ships with the bundled SBML model
[BIOMD0000000012](https://www.ebi.ac.uk/biomodels/BIOMD0000000012) — the Elowitz
& Leibler 2000 Repressilator.

Load a composite by name:

```python
from viva_copasi.composites import build_composite, list_composite_specs

print(list_composite_specs())   # ['steady-state', 'utc-process', 'utc-step']
sim = build_composite('utc-process')
```

Override the model or parameters:

```python
sim = build_composite('utc-step', overrides={
    'model_source': '/path/to/my_model.xml',
    'time': 50.0,
    'n_points': 51,
})
```

## API

### `CopasiUTCStep` (Step)

One-shot UTC trajectory.

Config: `model_source` (str), `time` (float, default 1.0), `n_points` (int, default 2),
`species_units` (`'concentration'` | `'amount'`, default `'concentration'`).

`species_units` sets the units of the default species output: `'concentration'`
(the prior behavior) or `'amount'` (particle number). It hides COPASI's bare-name
(particle count) vs `[name]` (concentration) split so every species is reported
consistently. Explicit `selections` are reported verbatim and are unaffected.

Outputs: `result` — dict with keys `time` (list), `columns` (list), `values` (list of rows).

### `CopasiUTCProcess` (Process)

Incremental UTC simulation.

Config: `model_source` (str), `time` (float, default 1.0), `intervals` (int, default 10),
`species_units` (`'concentration'` | `'amount'`, default `'concentration'`).

Outputs: `species_concentrations` (map[float]), `fluxes` (map[float]), `time` (list[float]).

### `CopasiSteadyStateStep` (Step)

Steady-state solve.

Config: `model_source` (str),
`species_units` (`'concentration'` | `'amount'`, default `'concentration'`).

Outputs: `results` — dict with keys `time`, `species_concentrations`, `fluxes`.

### `register_copasi(core=None)`

Register all COPASI process classes, the RAM emitter, and the
`SpeciesConcentrationsPlot` visualization into a bigraph core.

### `make_copasi_utc_document(...)` / `make_copasi_utc_process_document(...)` / `make_copasi_steady_state_document(...)`

Programmatic composite-document factories for callers that want full control
over wiring without a YAML file.

## Tests

```bash
python -m pytest tests/ -q
```

## Demos

```bash
python demo/demo_report.py       # runs all 3 classes; saves demo/report.html
python demo/composite_report.py  # runs utc-process composite; saves demo/utc-process-report.html
```
